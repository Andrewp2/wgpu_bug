
@group(0) @binding(0)
var<storage, read> numbers_input: array<u32>;

@group(0) @binding(1)
var<storage, read_write> numbers_output: array<u32>;

var<workgroup> shared_atomic: array<atomic<u32>, 64>;
var<workgroup> shared_workgroup_memory: array<u32, 64>;

fn prefix_sum_swap(wid: u32, lo: u32, hi: u32) {
    let before = shared_workgroup_memory[wid + lo];
    let after = shared_workgroup_memory[wid + hi];
    shared_workgroup_memory[wid + lo] = after;
    shared_workgroup_memory[wid + hi] += before;
}

fn prefix_sum_block_exclusive(wid: u32) {
    for (var i: u32 = 1u; i < 64u; i = i << 1u) {
        workgroupBarrier();
        if wid % (2u * i) == 0u {
            shared_workgroup_memory[wid + (2u * i) - 1u] += shared_workgroup_memory[wid + i - 1u];
        }
    }
    workgroupBarrier();
    // special case for first iteration
    if wid % 64u == 0u {
        // 64 / 2 - 1 = 31
        let before = shared_workgroup_memory[(64u / 2u) - 1u];

        shared_workgroup_memory[(64u / 2u) - 1u] = 0u;
        shared_workgroup_memory[64u - 1u] = before;
    }
    workgroupBarrier();
    // 32 16 8 4 2
    for (var i: u32 = 64u / 2u; i > 1u; i = i >> 1u) {
        workgroupBarrier();
        if wid % i == 0u {
            prefix_sum_swap(wid, (i / 2u) - 1u, i - 1u);
        }
    }
    workgroupBarrier();
}


@compute @workgroup_size(256, 1, 1)
fn prefix_sum(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(local_invocation_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let id = invocation_id.x;
    let wid = workgroup_id.x;

    let digit = numbers_input[id] & 63u;

    atomicAdd(&shared_atomic[digit], 1u);
    workgroupBarrier();

    if wid < 64u {
        let b = atomicLoad(&shared_atomic[wid]);
        workgroupBarrier();
        shared_workgroup_memory[wid] = b;
        workgroupBarrier();
        prefix_sum_block_exclusive(wid);
    }

    numbers_output[id] = shared_workgroup_memory[digit];
    //numbers_output[id] = shared_workgroup_memory[wid % 64u];
}