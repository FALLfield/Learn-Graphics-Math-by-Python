import taichi as ti
import numpy as np

# 初始化Taichi，设置为CPU架构
# arch可以设置为ti.cpu或ti.gpu，取决于硬件支持
ti.init(arch=ti.cpu)

# 定义屏幕分辨率
res_x, res_y = 800, 800
# 定义坐标系的界限
x_min, x_max = -1, 7
y_min, y_max = -1, 7
# 计算坐标变换因子 - 将世界坐标转换为屏幕坐标
scale_x = res_x / (x_max - x_min)
scale_y = res_y / (y_max - y_min)

# 定义向量作为Taichi字段
# shape=()表示这是标量字段，每个字段只包含一个向量
vec1 = ti.Vector.field(2, dtype=ti.f32, shape=())  # 第一个向量
vec2 = ti.Vector.field(2, dtype=ti.f32, shape=())  # 第二个向量
vec_sum = ti.Vector.field(2, dtype=ti.f32, shape=())  # 向量和

# 定义变换后向量的起点
transformed_vec1_origin = ti.Vector.field(2, dtype=ti.f32, shape=())
transformed_vec2_origin = ti.Vector.field(2, dtype=ti.f32, shape=())

# 计算向量加法的Taichi内核函数
@ti.kernel
def add_vectors():
    # 设置向量值
    vec1[None] = ti.Vector([2.0, 3.0])  # 向量1 [2, 3]
    vec2[None] = ti.Vector([3.0, 1.0])  # 向量2 [3, 1]
    
    # 计算向量和: vec_sum = vec1 + vec2
    vec_sum[None] = vec1[None] + vec2[None]
    
    # 设置变换后向量的起点
    transformed_vec1_origin[None] = ti.Vector([3.0, 1.0])
    transformed_vec2_origin[None] = ti.Vector([2.0, 3.0])

# 世界坐标转屏幕坐标函数
@ti.func
def world_to_screen(x, y):
    screen_x = (x - x_min) * scale_x
    screen_y = (y - y_min) * scale_y
    return screen_x, screen_y

# 绘制向量函数
@ti.func
def draw_vector(canvas, origin_x, origin_y, vec_x, vec_y, color):
    # 绘制向量线段
    start_x, start_y = world_to_screen(origin_x, origin_y)
    end_x, end_y = world_to_screen(origin_x + vec_x, origin_y + vec_y)
    canvas.line(ti.Vector([start_x, start_y]), ti.Vector([end_x, end_y]), color, 2.0)
    
    # 绘制箭头
    arrow_length = 10.0
    arrow_width = 4.0
    angle = ti.atan2(end_y - start_y, end_x - start_x)
    canvas.triangle(
        ti.Vector([end_x, end_y]),
        ti.Vector([end_x - arrow_length * ti.cos(angle - 0.3), end_y - arrow_length * ti.sin(angle - 0.3)]),
        ti.Vector([end_x - arrow_length * ti.cos(angle + 0.3), end_y - arrow_length * ti.sin(angle + 0.3)]),
        color
    )

# 绘制虚线向量函数
@ti.func
def draw_dashed_vector(canvas, origin_x, origin_y, vec_x, vec_y, color, dash_length):
    # 获取屏幕坐标起点和终点
    start_x, start_y = world_to_screen(origin_x, origin_y)
    end_x, end_y = world_to_screen(origin_x + vec_x, origin_y + vec_y)
    
    # 计算向量总长度和方向
    total_length = ti.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    dir_x, dir_y = (end_x - start_x) / total_length, (end_y - start_y) / total_length
    
    # 绘制虚线段
    num_segments = int(total_length / (dash_length * 2))
    for i in range(num_segments):
        seg_start_x = start_x + i * dash_length * 2 * dir_x
        seg_start_y = start_y + i * dash_length * 2 * dir_y
        seg_end_x = seg_start_x + dash_length * dir_x
        seg_end_y = seg_start_y + dash_length * dir_y
        canvas.line(ti.Vector([seg_start_x, seg_start_y]), ti.Vector([seg_end_x, seg_end_y]), color, 2.0)

# 绘制文本标签函数
@ti.func
def draw_grid(canvas):
    # 绘制坐标轴
    x_axis_start_x, x_axis_start_y = world_to_screen(x_min, 0)
    x_axis_end_x, x_axis_end_y = world_to_screen(x_max, 0)
    canvas.line(ti.Vector([x_axis_start_x, x_axis_start_y]), 
                ti.Vector([x_axis_end_x, x_axis_end_y]), 
                ti.Vector([0.5, 0.5, 0.5]), 
                1.0)
    
    y_axis_start_x, y_axis_start_y = world_to_screen(0, y_min)
    y_axis_end_x, y_axis_end_y = world_to_screen(0, y_max)
    canvas.line(ti.Vector([y_axis_start_x, y_axis_start_y]), 
                ti.Vector([y_axis_end_x, y_axis_end_y]), 
                ti.Vector([0.5, 0.5, 0.5]), 
                1.0)
    
    # 绘制网格线
    grid_spacing = 1.0
    for i in range(int(x_min), int(x_max)+1):
        if i == 0:  # 跳过坐标轴
            continue
        grid_x, grid_y_start = world_to_screen(i, y_min)
        grid_x, grid_y_end = world_to_screen(i, y_max)
        canvas.line(ti.Vector([grid_x, grid_y_start]), 
                    ti.Vector([grid_x, grid_y_end]), 
                    ti.Vector([0.3, 0.3, 0.3]), 
                    0.5)
    
    for j in range(int(y_min), int(y_max)+1):
        if j == 0:  # 跳过坐标轴
            continue
        grid_x_start, grid_y = world_to_screen(x_min, j)
        grid_x_end, grid_y = world_to_screen(x_max, j)
        canvas.line(ti.Vector([grid_x_start, grid_y]), 
                    ti.Vector([grid_x_end, grid_y]), 
                    ti.Vector([0.3, 0.3, 0.3]), 
                    0.5)

# 主要绘图函数
@ti.kernel
def render(canvas: ti.template()):
    # 绘制网格和坐标轴
    draw_grid(canvas)
    
    # 颜色定义
    red = ti.Vector([1.0, 0.0, 0.0])  # 向量1的颜色
    blue = ti.Vector([0.0, 0.0, 1.0])  # 向量2的颜色
    green = ti.Vector([0.0, 0.8, 0.0])  # 向量和的颜色
    
    # 绘制原始向量
    draw_vector(canvas, 0, 0, vec1[None][0], vec1[None][1], red)
    draw_vector(canvas, 0, 0, vec2[None][0], vec2[None][1], blue)
    
    # 绘制向量和
    draw_vector(canvas, 0, 0, vec_sum[None][0], vec_sum[None][1], green)
    
    # 绘制移动后的向量
    # 向量1移动到向量2的终点
    draw_vector(canvas, vec2[None][0], vec2[None][1], vec1[None][0], vec1[None][1], red * 0.7)
    
    # 向量2移动到向量1的终点
    draw_vector(canvas, vec1[None][0], vec1[None][1], vec2[None][0], vec2[None][1], blue * 0.7)
    
    # 绘制从变换向量起点连接到原点的虚线
    draw_dashed_vector(canvas, 0, 0, transformed_vec1_origin[None][0], transformed_vec1_origin[None][1], red * 0.5, 5)
    draw_dashed_vector(canvas, 0, 0, transformed_vec2_origin[None][0], transformed_vec2_origin[None][1], blue * 0.5, 5)
    
    # 绘制从变换向量起点到向量和终点的虚线
    draw_dashed_vector(canvas, transformed_vec1_origin[None][0], transformed_vec1_origin[None][1], 
                     vec_sum[None][0] - transformed_vec1_origin[None][0], 
                     vec_sum[None][1] - transformed_vec1_origin[None][1], green * 0.7, 5)
    
    draw_dashed_vector(canvas, transformed_vec2_origin[None][0], transformed_vec2_origin[None][1], 
                     vec_sum[None][0] - transformed_vec2_origin[None][0], 
                     vec_sum[None][1] - transformed_vec2_origin[None][1], green * 0.7, 5)

# 运行主程序
def main():
    # 执行向量加法计算
    add_vectors()
    
    # 创建Taichi GUI窗口
    window = ti.ui.Window("Vector Addition Visualization", (res_x, res_y))
    canvas = window.get_canvas()
    
    # 获取向量结果（用于信息显示）
    v1 = np.array([vec1[None][0], vec1[None][1]])
    v2 = np.array([vec2[None][0], vec2[None][1]])
    v_sum = np.array([vec_sum[None][0], vec_sum[None][1]])
    
    # 控制台输出
    print(f"Vector 1: [{v1[0]}, {v1[1]}]")
    print(f"Vector 2: [{v2[0]}, {v2[1]}]")
    print(f"Vector Sum: [{v_sum[0]}, {v_sum[1]}]")
    
    # 主循环
    while window.running:
        # 渲染场景
        canvas.clear(ti.Vector([1.0, 1.0, 1.0]))  # 白色背景
        render(canvas)
        
        # 添加文本信息
        window.GUI.begin("Vector Information", 0.01, 0.01, 0.3, 0.3)
        window.GUI.text(f"Vector 1: [{v1[0]:.1f}, {v1[1]:.1f}]")
        window.GUI.text(f"Vector 2: [{v2[0]:.1f}, {v2[1]:.1f}]")
        window.GUI.text(f"Vector Sum: [{v_sum[0]:.1f}, {v_sum[1]:.1f}]")
        window.GUI.text("Vectors can be added graphically")
        window.GUI.text("by placing them head-to-tail")
        window.GUI.end()
        
        # 显示帧
        window.show()

if __name__ == "__main__":
    main()

