//TODO 有必要吗？

// mut-mul
// 接收y, x, weight,y = x * weight + y,
// y，x 为行向量
pub struct MutMulArgs {
    pub beta: f32,
    pub alpha: f32,
    pub read_dst: bool,
}

// sum
// 接收out, y
// out = sum(y) 对第一个维度进行加法
