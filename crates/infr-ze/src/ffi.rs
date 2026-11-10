//! The Level Zero C API surface `infr-ze` uses, declared by hand from `ze_api.h` (loader 1.32)
//! and resolved at run time from `libze_loader` — the way `ash` opens the Vulkan loader, so
//! nothing links against oneAPI at build time.
//!
//! Only the entry points and structures this crate calls are declared. Every struct is a
//! `#[repr(C)]` mirror of its header definition (field order and types as declared there);
//! handles are opaque pointers.
#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_char, c_void};

pub type ze_result_t = u32;
pub const ZE_RESULT_SUCCESS: ze_result_t = 0;
pub const ZE_RESULT_NOT_READY: ze_result_t = 1;

pub type Handle = *mut c_void;

pub const ZE_INIT_FLAG_GPU_ONLY: u32 = 1;
pub const ZE_DEVICE_TYPE_GPU: u32 = 1;

pub const ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES: u32 = 0x3;
pub const ZE_STRUCTURE_TYPE_CONTEXT_DESC: u32 = 0xd;
pub const ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC: u32 = 0xe;
pub const ZE_STRUCTURE_TYPE_EVENT_POOL_DESC: u32 = 0x10;
pub const ZE_STRUCTURE_TYPE_EVENT_DESC: u32 = 0x11;
pub const ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC: u32 = 0x15;
pub const ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC: u32 = 0x16;
pub const ZE_STRUCTURE_TYPE_MODULE_DESC: u32 = 0x1b;
pub const ZE_STRUCTURE_TYPE_KERNEL_DESC: u32 = 0x1d;
pub const ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES: u32 = 0x10008;

pub const ZE_MODULE_FORMAT_IL_SPIRV: u32 = 0;
pub const ZE_MODULE_FORMAT_NATIVE: u32 = 1;
pub const ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS: u32 = 2;
pub const ZE_EVENT_POOL_FLAG_HOST_VISIBLE: u32 = 1;
pub const ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP: u32 = 4;
pub const ZE_EVENT_SCOPE_FLAG_HOST: u32 = 4;

#[repr(C)]
pub struct ze_device_properties_t {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub type_: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub flags: u32,
    pub subdevice_id: u32,
    pub core_clock_rate: u32,
    pub max_mem_alloc_size: u64,
    pub max_hardware_contexts: u32,
    pub max_command_queue_priority: u32,
    pub num_threads_per_eu: u32,
    pub physical_eu_simd_width: u32,
    pub num_eus_per_subslice: u32,
    pub num_subslices_per_slice: u32,
    pub num_slices: u32,
    pub timer_resolution: u64,
    pub timestamp_valid_bits: u32,
    pub kernel_timestamp_valid_bits: u32,
    pub uuid: [u8; 16],
    pub name: [c_char; 256],
}

#[repr(C)]
pub struct ze_pci_address_ext_t {
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
    pub function: u32,
}

#[repr(C)]
pub struct ze_pci_speed_ext_t {
    pub gen_version: i32,
    pub width: i32,
    pub max_bandwidth: i64,
}

#[repr(C)]
pub struct ze_pci_ext_properties_t {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub address: ze_pci_address_ext_t,
    pub max_speed: ze_pci_speed_ext_t,
}

#[repr(C)]
pub struct ze_context_desc_t {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
}

#[repr(C)]
pub struct ze_command_queue_desc_t {
    pub stype: u32,
    pub p_next: *const c_void,
    pub ordinal: u32,
    pub index: u32,
    pub flags: u32,
    pub mode: u32,
    pub priority: u32,
}

#[repr(C)]
pub struct ze_device_mem_alloc_desc_t {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub ordinal: u32,
}

#[repr(C)]
pub struct ze_host_mem_alloc_desc_t {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
}

#[repr(C)]
pub struct ze_module_desc_t {
    pub stype: u32,
    pub p_next: *const c_void,
    pub format: u32,
    pub input_size: usize,
    pub p_input_module: *const u8,
    pub p_build_flags: *const c_char,
    pub p_constants: *const c_void,
}

#[repr(C)]
pub struct ze_kernel_desc_t {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub p_kernel_name: *const c_char,
}

#[repr(C)]
pub struct ze_group_count_t {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[repr(C)]
pub struct ze_event_pool_desc_t {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub count: u32,
}

#[repr(C)]
pub struct ze_event_desc_t {
    pub stype: u32,
    pub p_next: *const c_void,
    pub index: u32,
    pub signal: u32,
    pub wait: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ze_kernel_timestamp_data_t {
    pub kernel_start: u64,
    pub kernel_end: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ze_kernel_timestamp_result_t {
    pub global: ze_kernel_timestamp_data_t,
    pub context: ze_kernel_timestamp_data_t,
}

macro_rules! ze_fns {
    ($( $name:ident : fn( $($arg:ty),* ) ; )*) => {
        /// The resolved entry points (`ze*` symbols of `libze_loader`).
        pub struct Fns {
            $( pub $name: unsafe extern "C" fn($($arg),*) -> ze_result_t, )*
        }
        impl Fns {
            /// Resolve every entry point; the first missing symbol is the error.
            ///
            /// # Safety
            /// `lib` must be a Level Zero loader whose symbols have the declared signatures.
            pub unsafe fn load(lib: &libloading::Library) -> Result<Self, String> {
                Ok(Self {
                    $( $name: *lib
                        .get::<unsafe extern "C" fn($($arg),*) -> ze_result_t>(
                            concat!(stringify!($name), "\0").as_bytes(),
                        )
                        .map_err(|e| format!("{}: {e}", stringify!($name)))?, )*
                })
            }
        }
    };
}

ze_fns! {
    zeInit: fn(u32);
    zeDriverGet: fn(*mut u32, *mut Handle);
    zeDeviceGet: fn(Handle, *mut u32, *mut Handle);
    zeDeviceGetProperties: fn(Handle, *mut ze_device_properties_t);
    zeDevicePciGetPropertiesExt: fn(Handle, *mut ze_pci_ext_properties_t);
    zeContextCreate: fn(Handle, *const ze_context_desc_t, *mut Handle);
    zeContextDestroy: fn(Handle);
    zeCommandListCreateImmediate: fn(Handle, Handle, *const ze_command_queue_desc_t, *mut Handle);
    zeCommandListDestroy: fn(Handle);
    zeCommandListHostSynchronize: fn(Handle, u64);
    zeCommandListAppendBarrier: fn(Handle, Handle, u32, *mut Handle);
    zeCommandListAppendMemoryCopy: fn(Handle, *mut c_void, *const c_void, usize, Handle, u32, *mut Handle);
    zeCommandListAppendLaunchKernel: fn(Handle, Handle, *const ze_group_count_t, Handle, u32, *mut Handle);
    zeMemAllocDevice: fn(Handle, *const ze_device_mem_alloc_desc_t, usize, usize, Handle, *mut *mut c_void);
    zeMemAllocHost: fn(Handle, *const ze_host_mem_alloc_desc_t, usize, usize, *mut *mut c_void);
    zeMemFree: fn(Handle, *mut c_void);
    zeModuleCreate: fn(Handle, Handle, *const ze_module_desc_t, *mut Handle, *mut Handle);
    zeModuleDestroy: fn(Handle);
    zeModuleBuildLogGetString: fn(Handle, *mut usize, *mut c_char);
    zeModuleBuildLogDestroy: fn(Handle);
    zeKernelCreate: fn(Handle, *const ze_kernel_desc_t, *mut Handle);
    zeKernelDestroy: fn(Handle);
    zeKernelSetGroupSize: fn(Handle, u32, u32, u32);
    zeKernelSetArgumentValue: fn(Handle, u32, usize, *const c_void);
    zeEventPoolCreate: fn(Handle, *const ze_event_pool_desc_t, u32, *mut Handle, *mut Handle);
    zeEventPoolDestroy: fn(Handle);
    zeEventCreate: fn(Handle, *const ze_event_desc_t, *mut Handle);
    zeEventDestroy: fn(Handle);
    zeEventHostReset: fn(Handle);
    zeEventQueryKernelTimestamp: fn(Handle, *mut ze_kernel_timestamp_result_t);
}
