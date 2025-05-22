import taichi as ti
import numpy as np
import math

# 初始化Taichi，设置为CPU架构
# arch可以设置为ti.cpu或ti.gpu，取决于硬件支持
ti.init(arch=ti.cpu)

# 定义屏幕分辨率
res_x, res_y = 800, 800
# 定义坐标系的界限
x_min, x_max = -2, 8
y_min, y_max = -2, 8
# 计算坐标变换因子 - 将世界坐标转换为屏幕坐标
scale_x = res_x / (x_max - x_min)
scale_y = res_y / (y_max - y_min)

# 定义向量作为Taichi字段
# shape=()表示这是标量字段，每个字段只包含一个向量
vec1 = ti.Vector.field(2, dtype=ti.f32, shape=())  # 第一个向量
vec2 = ti.Vector.field(2, dtype=ti.f32, shape=())  # 第二个向量
projection = ti.Vector.field(2, dtype=ti.f32, shape=())  # 投影向量
perpendicular = ti.Vector.field(2, dtype=ti.f32, shape=())  # 垂直分量向量

# 定义标量字段用于各种计算
dot_product = ti.field(dtype=ti.f32, shape=())  # 点积结果
norm_v1 = ti.field(dtype=ti.f32, shape=())  # 向量1的模长
norm_v2 = ti.field(dtype=ti.f32, shape=())  # 向量2的模长
cos_angle = ti.field(dtype=ti.f32, shape=())  # 夹角余弦值
angle_radians = ti.field(dtype=ti.f32, shape=())  # 夹角弧度值

# 定义Taichi内核函数来计算夹角和投影
@ti.kernel
def calculate_angle_and_projection():
    # 设置向量值
    vec1[None] = ti.Vector([3.0, 2.0])  # 向量1 [3, 2]
    vec2[None] = ti.Vector([2.0, 4.0])  # 向量2 [2, 4]
    
    # 计算点积: vec1·vec2 = vec1[0]*vec2[0] + vec1[1]*vec2[1]
    dot_product[None] = vec1[None].dot(vec2[None])
    
    # 计算向量的模长: |vec| = √(vec[0]² + vec[1]²)
    norm_v1[None] = vec1[None].norm()
    norm_v2[None] = vec2[None].norm()
    
    # 根据点积公式计算夹角的余弦值: cos(θ) = (vec1·vec2)/(|vec1|·|vec2|)
    # vec1·vec2是两个向量的点积，|vec1|和|vec2|是两个向量的模长
    # 该公式源自点积的几何定义: vec1·vec2 = |vec1|·|vec2|·cos(θ)
    cos_angle[None] = dot_product[None] / (norm_v1[None] * norm_v2[None])
    
    # 使用反余弦函数(arccos)从余弦值计算角度
    # ti.acos函数会自动处理边界情况
    angle_radians[None] = ti.acos(cos_angle[None])
    
    # 计算向量1在向量2上的投影
    # 投影长度 = (vec1·vec2) / |vec2|
    projection_length = dot_product[None] / norm_v2[None]
    # 投影向量 = (投影长度 / |vec2|) * vec2
    projection[None] = (projection_length / norm_v2[None]) * vec2[None]
    
    # 计算垂直分量: 向量1 = 投影向量 + 垂直分量
    perpendicular[None] = vec1[None] - projection[None]

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

# 绘制圆弧函数
@ti.func
def draw_arc(canvas, center_x, center_y, radius, start_angle, end_angle, color, num_segments):
    center_screen_x, center_screen_y = world_to_screen(center_x, center_y)
    radius_screen = radius * scale_x  # 假设x和y的缩放比例相同
    
    # 绘制圆弧的每一段
    for i in range(num_segments):
        angle1 = start_angle + i * (end_angle - start_angle) / num_segments
        angle2 = start_angle + (i + 1) * (end_angle - start_angle) / num_segments
        
        x1 = center_screen_x + radius_screen * ti.cos(angle1)
        y1 = center_screen_y + radius_screen * ti.sin(angle1)
        x2 = center_screen_x + radius_screen * ti.cos(angle2)
        y2 = center_screen_y + radius_screen * ti.sin(angle2)
        
        canvas.line(ti.Vector([x1, y1]), ti.Vector([x2, y2]), color, 2.0)

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
    green = ti.Vector([0.0, 0.8, 0.0])  # 投影向量的颜色
    purple = ti.Vector([0.8, 0.0, 0.8])  # 垂直分量的颜色
    gray = ti.Vector([0.5, 0.5, 0.5])  # 圆弧颜色
    
    # 绘制原始向量
    draw_vector(canvas, 0, 0, vec1[None][0], vec1[None][1], red)
    draw_vector(canvas, 0, 0, vec2[None][0], vec2[None][1], blue)
    
    # 绘制投影向量
    draw_vector(canvas, 0, 0, projection[None][0], projection[None][1], green)
    
    # 绘制垂直分量
    draw_vector(canvas, 0, 0, perpendicular[None][0], perpendicular[None][1], purple)
    
    # 绘制虚线连接
    draw_dashed_vector(canvas, 0, 0, vec1[None][0], vec1[None][1], red * 0.7, 5)
    draw_dashed_vector(canvas, 0, 0, projection[None][0], projection[None][1], green * 0.7, 5)
    draw_dashed_vector(canvas, 0, 0, perpendicular[None][0], perpendicular[None][1], purple * 0.7, 5)
    
    # 绘制角度圆弧
    arc_radius = 0.5
    draw_arc(canvas, 0, 0, arc_radius, 0, angle_radians[None], gray, 20)

# 运行主程序
def main():
    # 执行计算
    calculate_angle_and_projection()
    
    # 获取计算结果（用于控制台输出）
    angle_degrees = float(angle_radians[None]) * 180.0 / math.pi
    dot_prod = float(dot_product[None])
    similarity = float(cos_angle[None])
    proj_np = np.array([projection[None][0], projection[None][1]])
    
    # 控制台输出结果
    print(f"Angle between vectors: {angle_degrees:.2f} degrees")
    print(f"\nProjection of v1 onto v2: [{proj_np[0]:.2f}, {proj_np[1]:.2f}]")
    print(f"\nDirection Analysis:")
    print(f"Dot product: {dot_prod:.2f}")
    print(f"Similarity (cosine): {similarity:.2f}")
    
    if similarity > 0:
        print("Vectors are pointing in similar directions")
        if similarity > 0.9:
            print("Vectors are almost parallel")
    elif similarity < 0:
        print("Vectors are pointing in opposite directions")
        if similarity < -0.9:
            print("Vectors are almost antiparallel")
    else:
        print("Vectors are perpendicular")
    
    # 创建Taichi GUI窗口
    window = ti.ui.Window("Vector Dot Product Visualization", (res_x, res_y))
    canvas = window.get_canvas()
    
    # 主循环
    while window.running:
        # 渲染场景
        canvas.clear(ti.Vector([1.0, 1.0, 1.0]))  # 白色背景
        render(canvas)
        
        # 添加文本信息
        window.GUI.begin("Vector Information", 0.01, 0.01, 0.3, 0.3)
        window.GUI.text(f"Angle: {angle_degrees:.2f}°")
        window.GUI.text(f"Dot Product: {dot_prod:.2f}")
        window.GUI.text(f"Similarity: {similarity:.2f}")
        window.GUI.end()
        
        # 显示帧
        window.show()

if __name__ == "__main__":
    main()