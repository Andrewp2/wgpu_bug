use std::borrow::Cow;

use encase::{private::WriteInto, ShaderType, StorageBuffer};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    *,
};

fn main() {
    pollster::block_on(run());
}

/// Creates a GPU buffer from type T.
fn create_buffer<T: ShaderType + WriteInto>(
    t: T,
    device: &Device,
    label_str: &str,
    usage: BufferUsages,
) -> Buffer {
    let buf: Vec<u8> = Vec::new();
    // This looks strange, but is actually the way Bevy internally calculates its buffers.
    let mut x: StorageBuffer<Vec<u8>> = StorageBuffer::new(buf);
    x.write(&t).unwrap();
    let final_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label_str),
        contents: x.as_ref(),
        usage,
    });
    final_buffer
}

fn copy_buffer_to_buffer(encoder: &mut CommandEncoder, source: &Buffer, destination: &Buffer) {
    encoder.copy_buffer_to_buffer(source, 0, destination, 0, source.size());
}

async fn run() {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::all(),
        dx12_shader_compiler: Default::default(),
    });
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&Default::default(), None)
        .await
        .unwrap();
    let info = adapter.get_info();
    log::info!("backend: {:?}", info.backend);
    device.on_uncaptured_error(Box::new(move |error| {
        log::error!("{}", &error);
        panic!(
            "wgpu error (handling all wgpu errors as fatal):\n{:?}\n{:?}",
            &error, &info,
        );
    }));

    let mut rng_gen = ChaCha8Rng::seed_from_u64(2);
    let numbers: Vec<u32> = (0..256).map(|_| rng_gen.gen()).collect();
    let numbers_copy = numbers.clone();

    let buffer = create_buffer(
        numbers,
        &device,
        "numbers buffer",
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );

    let buffer_2 = device.create_buffer(&BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader_module = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("prefix_sum.wgsl"))),
    });

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("bind group layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: None,
        layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        })),
        module: &shader_module,
        entry_point: "prefix_sum",
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: buffer_2.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    {
        let compute_pass_desc = ComputePassDescriptor {
            label: Some("compute pass"),
        };
        let mut compute_pass = encoder.begin_compute_pass(&compute_pass_desc);
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }
    copy_buffer_to_buffer(&mut encoder, &buffer_2, &readback_buffer);
    queue.submit(Some(encoder.finish()));

    {
        let readback_slice = readback_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        readback_slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(MaintainBase::Wait);

        if let Some(Ok(())) = rx.receive().await {
            let readback_data = readback_slice.get_mapped_range();
            let readback: Vec<u32> = bytemuck::cast_slice(&readback_data).to_vec();
            drop(readback_data);
            readback_buffer.unmap();

            let mut histogram: Vec<u32> = vec![0; 64];
            numbers_copy
                .iter()
                .for_each(|x| histogram[(*x & 63) as usize] += 1);

            let prefix_sum: Vec<u32> = histogram
                .iter()
                .scan(0, |state, elem| {
                    *state += *elem;
                    Some(*state - *elem)
                })
                .collect();

            for (i, val) in readback.iter().enumerate() {
                println!(
                    "i: {}, cpu: {}, gpu: {}",
                    i,
                    prefix_sum[(numbers_copy[i] & 63) as usize],
                    val
                );
            }
        }
    }
}
