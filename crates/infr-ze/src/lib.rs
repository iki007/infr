//! `infr-ze`: a native Intel GPU backend through **Level Zero** (experimental, `docs/backlog.md`
//! B102). Level Zero is opened with `dlopen` at run time (no oneAPI at build time), and kernels are
//! OpenCL C compiled offline by Intel's `ocloc` into native binaries embedded in the crate
//! (`build.rs`) — the Level Zero analogues of `ash` and `glslc` in `infr-vulkan`. OpenCL C reaches
//! the Xe2 primitives the Vulkan path cannot: DPAS (`intel_sub_group_*_matrix_mad_*`) and 2D block
//! loads/stores (`intel_sub_group_2d_block_*`).
//!
//! What exists so far is the device layer — driver/device discovery, a context with one immediate
//! command list, device memory, modules, kernels, kernel timestamps — and the probes that measure
//! what this path buys (`tests/probe.rs`).
pub mod ffi;
pub mod kernels;

use ffi::*;
use std::ffi::{c_void, CStr, CString};
use std::rc::Rc;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Level Zero loader not available: {0}")]
    Load(String),
    #[error("{call} failed: ze_result 0x{code:x}")]
    Call { call: &'static str, code: u32 },
    #[error("no Level Zero GPU matches {0:?}")]
    NoDevice(String),
    #[error("module build failed: {0}")]
    Build(String),
}

pub type Result<T> = std::result::Result<T, Error>;

fn check(call: &'static str, code: ze_result_t) -> Result<()> {
    if code == ZE_RESULT_SUCCESS {
        Ok(())
    } else {
        Err(Error::Call { call, code })
    }
}

macro_rules! ze {
    ($f:expr, $name:ident ( $($arg:expr),* )) => {
        check(stringify!($name), unsafe { ($f.$name)($($arg),*) })
    };
}

/// The opened loader and its entry points.
pub struct Ze {
    f: Fns,
    _lib: libloading::Library,
}

impl Ze {
    /// Open the system Level Zero loader and initialize it for GPUs.
    pub fn open() -> Result<Arc<Ze>> {
        let name = if cfg!(windows) {
            "ze_loader.dll"
        } else {
            "libze_loader.so.1"
        };
        // SAFETY: loading the vendor loader runs its initializers, as any Level Zero client does.
        let lib =
            unsafe { libloading::Library::new(name) }.map_err(|e| Error::Load(e.to_string()))?;
        // SAFETY: the symbols are the Level Zero API with the signatures declared in `ffi`.
        let f = unsafe { Fns::load(&lib) }.map_err(Error::Load)?;
        ze!(f, zeInit(ZE_INIT_FLAG_GPU_ONLY))?;
        Ok(Arc::new(Ze { f, _lib: lib }))
    }

    /// Every GPU device of every driver.
    pub fn devices(&self) -> Result<Vec<DeviceInfo>> {
        let mut n = 0u32;
        ze!(self.f, zeDriverGet(&mut n, std::ptr::null_mut()))?;
        let mut drivers = vec![std::ptr::null_mut(); n as usize];
        ze!(self.f, zeDriverGet(&mut n, drivers.as_mut_ptr()))?;
        let mut out = Vec::new();
        for &driver in &drivers {
            let mut nd = 0u32;
            ze!(self.f, zeDeviceGet(driver, &mut nd, std::ptr::null_mut()))?;
            let mut devs = vec![std::ptr::null_mut(); nd as usize];
            ze!(self.f, zeDeviceGet(driver, &mut nd, devs.as_mut_ptr()))?;
            for &device in &devs {
                // SAFETY: an all-zero bit pattern is valid for this plain-data struct.
                let mut p: ze_device_properties_t = unsafe { std::mem::zeroed() };
                p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
                ze!(self.f, zeDeviceGetProperties(device, &mut p))?;
                if p.type_ != ZE_DEVICE_TYPE_GPU {
                    continue;
                }
                // SAFETY: as above.
                let mut pci: ze_pci_ext_properties_t = unsafe { std::mem::zeroed() };
                pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
                let bdf = if unsafe { (self.f.zeDevicePciGetPropertiesExt)(device, &mut pci) }
                    == ZE_RESULT_SUCCESS
                {
                    let a = &pci.address;
                    format!("{:02x}:{:02x}.{:x}", a.bus, a.device, a.function)
                } else {
                    String::new()
                };
                // SAFETY: the driver NUL-terminates `name` within its 256 bytes.
                let name = unsafe { CStr::from_ptr(p.name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                out.push(DeviceInfo {
                    driver,
                    device,
                    name,
                    pci: bdf,
                    eus: p.num_slices * p.num_subslices_per_slice * p.num_eus_per_subslice,
                    clock_mhz: p.core_clock_rate,
                    timer_ns: p.timer_resolution,
                    ts_bits: p.kernel_timestamp_valid_bits,
                });
            }
        }
        Ok(out)
    }
}

/// One discovered GPU.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    driver: Handle,
    device: Handle,
    pub name: String,
    /// PCI `bus:device.function`, e.g. `03:00.0` (empty when the driver lacks the extension).
    pub pci: String,
    pub eus: u32,
    pub clock_mhz: u32,
    /// Kernel timestamp tick length in nanoseconds.
    pub timer_ns: u64,
    pub ts_bits: u32,
}

struct Inner {
    ze: Arc<Ze>,
    ctx: Handle,
    device: Handle,
    cl: Handle,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // SAFETY: the handles were created by this `Inner` and nothing outlives it (every child
        // object holds an `Rc<Inner>`).
        unsafe {
            (self.ze.f.zeCommandListDestroy)(self.cl);
            (self.ze.f.zeContextDestroy)(self.ctx);
        }
    }
}

/// An opened GPU: a context and one in-order immediate command list on the compute engine.
/// Single-threaded (`Rc`): a Level Zero command list needs external synchronization, and the
/// handles are raw pointers.
pub struct Gpu {
    inner: Rc<Inner>,
    pub info: DeviceInfo,
}

impl Gpu {
    /// Open the GPU whose PCI address is `pci` (`"03:00.0"`), or the first GPU for `None`.
    pub fn open(ze: &Arc<Ze>, pci: Option<&str>) -> Result<Gpu> {
        let devs = ze.devices()?;
        let info = match pci {
            Some(want) => devs.into_iter().find(|d| d.pci == want),
            None => devs.into_iter().next(),
        }
        .ok_or_else(|| Error::NoDevice(pci.unwrap_or("any").to_string()))?;
        let cdesc = ze_context_desc_t {
            stype: ZE_STRUCTURE_TYPE_CONTEXT_DESC,
            p_next: std::ptr::null(),
            flags: 0,
        };
        let mut ctx = std::ptr::null_mut();
        ze!(ze.f, zeContextCreate(info.driver, &cdesc, &mut ctx))?;
        // Ordinal 0 is the compute (render+compute) queue group on Intel GPUs.
        let qdesc = ze_command_queue_desc_t {
            stype: ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
            p_next: std::ptr::null(),
            ordinal: 0,
            index: 0,
            flags: 0,
            mode: ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
            priority: 0,
        };
        let mut cl = std::ptr::null_mut();
        if let Err(e) = ze!(
            ze.f,
            zeCommandListCreateImmediate(ctx, info.device, &qdesc, &mut cl)
        ) {
            // SAFETY: `ctx` was just created and is not shared.
            unsafe { (ze.f.zeContextDestroy)(ctx) };
            return Err(e);
        }
        Ok(Gpu {
            inner: Rc::new(Inner {
                ze: ze.clone(),
                ctx,
                device: info.device,
                cl,
            }),
            info,
        })
    }

    fn f(&self) -> &Fns {
        &self.inner.ze.f
    }

    /// Device-local memory of `bytes` bytes (64-byte aligned).
    pub fn alloc(&self, bytes: usize) -> Result<DevBuf> {
        let desc = ze_device_mem_alloc_desc_t {
            stype: ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            p_next: std::ptr::null(),
            flags: 0,
            ordinal: 0,
        };
        let mut ptr = std::ptr::null_mut();
        ze!(
            self.f(),
            zeMemAllocDevice(
                self.inner.ctx,
                &desc,
                bytes.max(1),
                64,
                self.inner.device,
                &mut ptr
            )
        )?;
        Ok(DevBuf {
            inner: self.inner.clone(),
            ptr,
            len: bytes,
        })
    }

    /// Copy `data` into `buf` (from offset 0) and wait for it.
    pub fn upload(&self, buf: &DevBuf, data: &[u8]) -> Result<()> {
        assert!(data.len() <= buf.len);
        ze!(
            self.f(),
            zeCommandListAppendMemoryCopy(
                self.inner.cl,
                buf.ptr,
                data.as_ptr() as *const c_void,
                data.len(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut()
            )
        )?;
        self.sync()
    }

    /// Copy `out.len()` bytes of `buf` (from offset 0) back to the host, after all prior work.
    pub fn download(&self, buf: &DevBuf, out: &mut [u8]) -> Result<()> {
        assert!(out.len() <= buf.len);
        ze!(
            self.f(),
            zeCommandListAppendMemoryCopy(
                self.inner.cl,
                out.as_mut_ptr() as *mut c_void,
                buf.ptr as *const c_void,
                out.len(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut()
            )
        )?;
        self.sync()
    }

    /// Wait until everything appended so far has executed.
    pub fn sync(&self) -> Result<()> {
        ze!(
            self.f(),
            zeCommandListHostSynchronize(self.inner.cl, u64::MAX)
        )
    }

    /// An execution barrier between the previous and the next appended commands.
    pub fn barrier(&self) -> Result<()> {
        ze!(
            self.f(),
            zeCommandListAppendBarrier(
                self.inner.cl,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut()
            )
        )
    }

    /// Load a native binary built by `ocloc` for this device.
    pub fn module(&self, binary: &[u8]) -> Result<Rc<Module>> {
        let desc = ze_module_desc_t {
            stype: ZE_STRUCTURE_TYPE_MODULE_DESC,
            p_next: std::ptr::null(),
            format: ZE_MODULE_FORMAT_NATIVE,
            input_size: binary.len(),
            p_input_module: binary.as_ptr(),
            p_build_flags: c"".as_ptr(),
            p_constants: std::ptr::null(),
        };
        let mut module = std::ptr::null_mut();
        let mut log = std::ptr::null_mut();
        // SAFETY: the descriptor points at live memory for the duration of the call.
        let code = unsafe {
            (self.f().zeModuleCreate)(
                self.inner.ctx,
                self.inner.device,
                &desc,
                &mut module,
                &mut log,
            )
        };
        let mut text = String::new();
        if !log.is_null() {
            let mut n = 0usize;
            // SAFETY: the two-call size/fill pattern of zeModuleBuildLogGetString.
            unsafe {
                (self.f().zeModuleBuildLogGetString)(log, &mut n, std::ptr::null_mut());
                let mut bytes = vec![0 as std::ffi::c_char; n.max(1)];
                (self.f().zeModuleBuildLogGetString)(log, &mut n, bytes.as_mut_ptr());
                text = CStr::from_ptr(bytes.as_ptr())
                    .to_string_lossy()
                    .into_owned();
                (self.f().zeModuleBuildLogDestroy)(log);
            }
        }
        if code != ZE_RESULT_SUCCESS {
            return Err(Error::Build(format!("ze_result 0x{code:x}: {text}")));
        }
        Ok(Rc::new(Module {
            inner: self.inner.clone(),
            handle: module,
        }))
    }

    /// Append a launch of `k` over `groups` workgroups.
    pub fn launch(&self, k: &Kernel, groups: [u32; 3]) -> Result<()> {
        self.launch_ev(k, groups, std::ptr::null_mut())
    }

    fn launch_ev(&self, k: &Kernel, groups: [u32; 3], ev: Handle) -> Result<()> {
        let gc = ze_group_count_t {
            x: groups[0],
            y: groups[1],
            z: groups[2],
        };
        ze!(
            self.f(),
            zeCommandListAppendLaunchKernel(
                self.inner.cl,
                k.handle,
                &gc,
                ev,
                0,
                std::ptr::null_mut()
            )
        )
    }

    /// Time `reps` back-to-back launches of `k` with kernel timestamps; returns the device time of
    /// each launch in nanoseconds (each launch is ordered after the previous one).
    pub fn time_launches(&self, k: &Kernel, groups: [u32; 3], reps: u32) -> Result<Vec<f64>> {
        let pdesc = ze_event_pool_desc_t {
            stype: ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
            p_next: std::ptr::null(),
            flags: ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP,
            count: reps,
        };
        let mut pool = std::ptr::null_mut();
        let mut dev = self.inner.device;
        ze!(
            self.f(),
            zeEventPoolCreate(self.inner.ctx, &pdesc, 1, &mut dev, &mut pool)
        )?;
        let mut evs = Vec::with_capacity(reps as usize);
        let mut res = Ok(());
        for i in 0..reps {
            let edesc = ze_event_desc_t {
                stype: ZE_STRUCTURE_TYPE_EVENT_DESC,
                p_next: std::ptr::null(),
                index: i,
                signal: ZE_EVENT_SCOPE_FLAG_HOST,
                wait: ZE_EVENT_SCOPE_FLAG_HOST,
            };
            let mut ev = std::ptr::null_mut();
            res = ze!(self.f(), zeEventCreate(pool, &edesc, &mut ev));
            if res.is_err() {
                break;
            }
            evs.push(ev);
        }
        let mut out = Vec::new();
        if res.is_ok() {
            for &ev in &evs {
                if let Err(e) = self.launch_ev(k, groups, ev).and_then(|_| self.barrier()) {
                    res = Err(e);
                    break;
                }
            }
        }
        if res.is_ok() {
            res = self.sync();
        }
        if res.is_ok() {
            let ns = self.info.timer_ns as f64;
            let mask = if self.info.ts_bits >= 64 {
                u64::MAX
            } else {
                (1u64 << self.info.ts_bits) - 1
            };
            for &ev in &evs {
                let mut t = ze_kernel_timestamp_result_t::default();
                if let Err(e) = ze!(self.f(), zeEventQueryKernelTimestamp(ev, &mut t)) {
                    res = Err(e);
                    break;
                }
                let d = t.global.kernel_end.wrapping_sub(t.global.kernel_start) & mask;
                out.push(d as f64 * ns);
            }
        }
        // SAFETY: the events and pool were created above and every command using them has
        // completed (or failed before submission).
        unsafe {
            for ev in evs {
                (self.f().zeEventDestroy)(ev);
            }
            (self.f().zeEventPoolDestroy)(pool);
        }
        res.map(|_| out)
    }
}

/// Device-local memory, freed on drop.
pub struct DevBuf {
    inner: Rc<Inner>,
    ptr: *mut c_void,
    pub len: usize,
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        // SAFETY: `ptr` was allocated in this context and is freed once.
        unsafe { (self.inner.ze.f.zeMemFree)(self.inner.ctx, self.ptr) };
    }
}

/// A loaded module.
pub struct Module {
    inner: Rc<Inner>,
    handle: Handle,
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: created by `Gpu::module`; kernels keep the module alive through `_module`.
        unsafe { (self.inner.ze.f.zeModuleDestroy)(self.handle) };
    }
}

impl Module {
    /// The kernel `name` of this module.
    pub fn kernel(self: &Rc<Self>, name: &str) -> Result<Kernel> {
        let cname = CString::new(name).expect("kernel name has no NUL");
        let desc = ze_kernel_desc_t {
            stype: ZE_STRUCTURE_TYPE_KERNEL_DESC,
            p_next: std::ptr::null(),
            flags: 0,
            p_kernel_name: cname.as_ptr(),
        };
        let mut handle = std::ptr::null_mut();
        ze!(
            self.inner.ze.f,
            zeKernelCreate(self.handle, &desc, &mut handle)
        )?;
        Ok(Kernel {
            _module: self.clone(),
            handle,
        })
    }
}

/// A kernel with its arguments and group size set on the object (Level Zero's model).
pub struct Kernel {
    _module: Rc<Module>,
    handle: Handle,
}

impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: created by `Module::kernel`.
        unsafe { (self._module.inner.ze.f.zeKernelDestroy)(self.handle) };
    }
}

impl Kernel {
    fn f(&self) -> &Fns {
        &self._module.inner.ze.f
    }

    pub fn group_size(&self, x: u32, y: u32, z: u32) -> Result<()> {
        ze!(self.f(), zeKernelSetGroupSize(self.handle, x, y, z))
    }

    /// Bind a device buffer (a global pointer argument).
    pub fn arg_buf(&self, index: u32, buf: &DevBuf) -> Result<()> {
        let p = buf.ptr;
        ze!(
            self.f(),
            zeKernelSetArgumentValue(
                self.handle,
                index,
                std::mem::size_of::<*mut c_void>(),
                &p as *const *mut c_void as *const c_void
            )
        )
    }

    /// Bind a by-value argument.
    pub fn arg<T: Copy>(&self, index: u32, v: T) -> Result<()> {
        ze!(
            self.f(),
            zeKernelSetArgumentValue(
                self.handle,
                index,
                std::mem::size_of::<T>(),
                &v as *const T as *const c_void
            )
        )
    }
}
