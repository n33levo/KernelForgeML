struct Params {
  size_m: u32,
  size_n: u32,
  size_k: u32,
  _padding: u32,
};

@group(0) @binding(0)
var<storage, read> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

fn lhs_index(m: u32, k: u32) -> u32 {
  return m * params.size_k + k;
}

fn rhs_index(k: u32, n: u32) -> u32 {
  return k * params.size_n + n;
}

fn out_index(m: u32, n: u32) -> u32 {
  return m * params.size_n + n;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let m = gid.x;
  let n = gid.y;
  if (m >= params.size_m || n >= params.size_n) {
    return;
  }

  var acc: f32 = 0.0;
  for (var k: u32 = 0u; k < params.size_k; k = k + 1u) {
    acc = acc + lhs[lhs_index(m, k)] * rhs[rhs_index(k, n)];
  }

  output[out_index(m, n)] = acc;
}
