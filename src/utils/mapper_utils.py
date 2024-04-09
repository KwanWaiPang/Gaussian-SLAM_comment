
import cv2
import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch


def compute_opt_views_distribution(keyframes_num, iterations_num, current_frame_iter) -> np.ndarray:
    """ Computes the probability distribution for selecting views based on the current iteration.
    Args:
        keyframes_num: The total number of keyframes.
        iterations_num: The total number of iterations planned.
        current_frame_iter: The current iteration number.
    Returns:
        An array representing the probability distribution of keyframes.
    """
    if keyframes_num == 1:
        return np.array([1.0])
    prob = np.full(keyframes_num, (iterations_num - current_frame_iter) / (keyframes_num - 1))
    prob[0] = current_frame_iter
    prob /= prob.sum()
    return prob

# 就是获取相机可视的3D范围的点
def compute_camera_frustum_corners(depth_map: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """ Computes the 3D coordinates of the camera frustum corners based on the depth map, pose, and intrinsics.
    Args:
        depth_map: The depth map of the scene.
        pose: The camera pose matrix.
        intrinsics: The camera intrinsic matrix.
    Returns:
        An array of 3D coordinates for the frustum corners.
    """
    # 首先，获取深度图的高度和宽度
    height, width = depth_map.shape

    # 对深度图进行处理，将深度值大于0的像素点提取出来，即去除了无效深度值（例如深度值为0的部分）。
    depth_map = depth_map[depth_map > 0]

    # 计算提取出的深度值的最小值和最大值，分别表示相机视锥体的最近点和最远点。
    min_depth, max_depth = depth_map.min(), depth_map.max()

    # 构建一个包含相机视锥体角点的初始数组corners，其中每一行对应一个角点的坐标。这里包含了视锥体的四个底面角点和四个顶面角点。
    corners = np.array(
        [
            [0, 0, min_depth],
            [width, 0, min_depth],
            [0, height, min_depth],
            [width, height, min_depth],
            [0, 0, max_depth],
            [width, 0, max_depth],
            [0, height, max_depth],
            [width, height, max_depth],
        ]
    )

    # 根据相机内参，通过逆投影计算角点的三维坐标。具体来说，通过将图像坐标转换为归一化平面坐标，然后再转换为相机坐标系下的三维坐标。这里使用了相机内参矩阵中的焦距和主点信息。
    x = (corners[:, 0] - intrinsics[0, 2]) * corners[:, 2] / intrinsics[0, 0]
    y = (corners[:, 1] - intrinsics[1, 2]) * corners[:, 2] / intrinsics[1, 1]
    z = corners[:, 2]

    # 将计算得到的三维角点坐标堆叠成一个矩阵，每一列代表一个角点的坐标，并添加齐次坐标的最后一维。
    corners_3d = np.vstack((x, y, z, np.ones(x.shape[0]))).T

    # 根据相机位姿，将角点从相机坐标系转换到世界坐标系。
    corners_3d = pose @ corners_3d.T

    # 最后，返回转换后的角点坐标，但是去除齐次坐标的最后一维，得到真实的三维坐标。
    return corners_3d.T[:, :3]


def compute_camera_frustum_planes(frustum_corners: np.ndarray) -> torch.Tensor:
    """ Computes the planes of the camera frustum from its corners.
    Args:
        frustum_corners: An array of 3D coordinates representing the corners of the frustum.

    Returns:
        A tensor of frustum planes.
    """
    # near, far, left, right, top, bottom
    planes = torch.stack(
        [
            torch.cross(
                frustum_corners[2] - frustum_corners[0],
                frustum_corners[1] - frustum_corners[0],
            ),
            torch.cross(
                frustum_corners[6] - frustum_corners[4],
                frustum_corners[5] - frustum_corners[4],
            ),
            torch.cross(
                frustum_corners[4] - frustum_corners[0],
                frustum_corners[2] - frustum_corners[0],
            ),
            torch.cross(
                frustum_corners[7] - frustum_corners[3],
                frustum_corners[1] - frustum_corners[3],
            ),
            torch.cross(
                frustum_corners[5] - frustum_corners[1],
                frustum_corners[3] - frustum_corners[1],
            ),
            torch.cross(
                frustum_corners[6] - frustum_corners[2],
                frustum_corners[0] - frustum_corners[2],
            ),
        ]
    )
    D = torch.stack([-torch.dot(plane, frustum_corners[i]) for i, plane in enumerate(planes)])
    return torch.cat([planes, D[:, None]], dim=1).float()


def compute_frustum_aabb(frustum_corners: torch.Tensor):
    """ Computes a mask indicating which points lie inside a given axis-aligned bounding box (AABB).
    Args:
        points: An array of 3D points.
        min_corner: The minimum corner of the AABB.
        max_corner: The maximum corner of the AABB.
    Returns:
        A boolean array indicating whether each point lies inside the AABB.
    """
    return torch.min(frustum_corners, axis=0).values, torch.max(frustum_corners, axis=0).values


def points_inside_aabb_mask(points: np.ndarray, min_corner: np.ndarray, max_corner: np.ndarray) -> np.ndarray:
    """ Computes a mask indicating which points lie inside the camera frustum.
    Args:
        points: A tensor of 3D points.
        frustum_planes: A tensor representing the planes of the frustum.
    Returns:
        A boolean tensor indicating whether each point lies inside the frustum.
    """
    return (
        (points[:, 0] >= min_corner[0])
        & (points[:, 0] <= max_corner[0])
        & (points[:, 1] >= min_corner[1])
        & (points[:, 1] <= max_corner[1])
        & (points[:, 2] >= min_corner[2])
        & (points[:, 2] <= max_corner[2]))


def points_inside_frustum_mask(points: torch.Tensor, frustum_planes: torch.Tensor) -> torch.Tensor:
    """ Computes a mask indicating which points lie inside the camera frustum.
    Args:
        points: A tensor of 3D points.
        frustum_planes: A tensor representing the planes of the frustum.
    Returns:
        A boolean tensor indicating whether each point lies inside the frustum.
    """
    num_pts = points.shape[0]
    ones = torch.ones(num_pts, 1).to(points.device)
    plane_product = torch.cat([points, ones], axis=1) @ frustum_planes.T
    return torch.all(plane_product <= 0, axis=1)

# 这个函数的目的是确定哪些点位于相机视锥体内，以便在进行后续的处理时，只对位于视锥体内的点进行计算，从而提高计算效率。
def compute_frustum_point_ids(pts: torch.Tensor, frustum_corners: torch.Tensor, device: str = "cuda"):
    """ Identifies points within the camera frustum, optimizing for computation on a specified device.
    Args:
        pts: A tensor of 3D points.
        frustum_corners: A tensor of 3D coordinates representing the corners of the frustum.
        device: The computation device ("cuda" or "cpu").
    Returns:
        Indices of points lying inside the frustum.（返回在视角范围内的点的索引）
    """

    # 如果输入的pts张量的行数为0，则直接返回一个空的张量
    if pts.shape[0] == 0:
        return torch.tensor([], dtype=torch.int64, device=device)
    # Broad phase
    pts = pts.to(device)
    frustum_corners = frustum_corners.to(device)

    min_corner, max_corner = compute_frustum_aabb(frustum_corners)
    inside_aabb_mask = points_inside_aabb_mask(pts, min_corner, max_corner)

    # Narrow phase
    frustum_planes = compute_camera_frustum_planes(frustum_corners)
    frustum_planes = frustum_planes.to(device)
    inside_frustum_mask = points_inside_frustum_mask(pts[inside_aabb_mask], frustum_planes)

    inside_aabb_mask[inside_aabb_mask == 1] = inside_frustum_mask
    return torch.where(inside_aabb_mask)[0]


def sample_pixels_based_on_gradient(image: np.ndarray, num_samples: int) -> np.ndarray:
    """ Samples pixel indices based on the gradient magnitude of an image.（根据梯度值来采样）
    Args:
        image: The image from which to sample pixels.
        num_samples: The number of pixels to sample.
    Returns:
        Indices of the sampled pixels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize the gradient magnitude to create a probability map
    prob_map = grad_magnitude / np.sum(grad_magnitude)

    # Flatten the probability map
    prob_map_flat = prob_map.flatten()

    # Sample pixel indices based on the probability map
    sampled_indices = np.random.choice(prob_map_flat.size, size=num_samples, p=prob_map_flat)
    return sampled_indices.T


def compute_new_points_ids(frustum_points: torch.Tensor, new_pts: torch.Tensor,
                           radius: float = 0.03, device: str = "cpu") -> torch.Tensor:
    """ Having newly initialized points, decides which of them should be added to the submap.
        For every new point, if there are no neighbors within the radius in the frustum points,
        it is added to the submap.
    Args:
        frustum_points: Point within a current frustum of the active submap of shape (N, 3)
        new_pts: New 3D Gaussian means which are about to be added to the submap of shape (N, 3)
        radius: Radius whithin which the points are considered to be neighbors
        device: Execution device
    Returns:
        Indicies of the new points that should be added to the submap of shape (N)
    """
    # 函数首先检查 frustum_points 是否为空，如果为空，则直接返回所有新点的索引，因为在当前视锥体中没有其他点，所有新点都应该添加到子地图中。
    if frustum_points.shape[0] == 0:
        return torch.arange(new_pts.shape[0])
    if device == "cpu":
        pts_index = faiss.IndexFlatL2(3)
    else:
        pts_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss.IndexFlatL2(3))
    # 将 frustum_points 和 new_pts 移动到指定的计算设备上，并将 frustum_points 添加到 Faiss 索引中。
    frustum_points = frustum_points.to(device)
    new_pts = new_pts.to(device)
    pts_index.add(frustum_points)

    # split函数，用于将一个张量沿着指定的维度（这里是0维，即按行）分割成多个小张量。
    #  将张量 new_pts 沿着行方向分割成多个小张量，每个小张量包含的行数是 65535 行（或者更少，如果 new_pts 的行数不是 65535 的倍数）。
    split_pos = torch.split(new_pts, 65535, dim=0)
    distances, ids = [], []
    # 函数遍历 new_pts
    for split_p in split_pos:
        # 使用 Faiss 索引搜索最近的8个邻居点，并计算它们与新点之间的距离。
        distance, id = pts_index.search(split_p.float(), 8)
        distances.append(distance)
        ids.append(id)
    distances = torch.cat(distances, dim=0)
    ids = torch.cat(ids, dim=0)
    # 统计每个新点周围的邻居数。
    neighbor_num = (distances < radius).sum(axis=1).int()
    pts_index.reset()
    # 最后，函数重置 Faiss 索引，并返回那些周围没有邻居的新点的索引，这些点应该被添加到子地图中。
    return torch.where(neighbor_num == 0)[0]


def rotation_to_euler(R: torch.Tensor) -> torch.Tensor:
    """
    Converts a rotation matrix to Euler angles.
    Args:
        R: A rotation matrix.
    Returns:
        Euler angles corresponding to the rotation matrix.
    """
    sy = torch.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
    else:
        x = torch.atan2(-R[1, 2], R[1, 1])
        y = torch.atan2(-R[2, 0], sy)
        z = 0

    return torch.tensor([x, y, z]) * (180 / np.pi)

# 获取两帧之间的相对位姿
def exceeds_motion_thresholds(current_c2w: torch.Tensor, last_submap_c2w: torch.Tensor,
                              rot_thre: float = 50, trans_thre: float = 0.5) -> bool:
    """  Checks if a camera motion exceeds certain rotation and translation thresholds
    Args:
        current_c2w: The current camera-to-world transformation matrix.
        last_submap_c2w: The last submap's camera-to-world transformation matrix.
        rot_thre: The rotation threshold for triggering a new submap.
        trans_thre: The translation threshold for triggering a new submap.

    Returns:
        A boolean indicating whether a new submap is required.
    """
    delta_pose = torch.matmul(torch.linalg.inv(last_submap_c2w).float(), current_c2w.float())
    translation_diff = torch.norm(delta_pose[:3, 3])
    rot_euler_diff_deg = torch.abs(rotation_to_euler(delta_pose[:3, :3]))
    exceeds_thresholds = (translation_diff > trans_thre) or torch.any(rot_euler_diff_deg > rot_thre)
    return exceeds_thresholds.item()

# 用于计算一个RGB图像的边缘掩码，采用了几何边缘的方法
def geometric_edge_mask(rgb_image: np.ndarray, dilate: bool = True, RGB: bool = False) -> np.ndarray:
    """ Computes an edge mask for an RGB image using geometric edges.
    Args:
        rgb_image: The RGB image.(是一个numpy数组，表示RGB图像。)
        dilate: Whether to dilate the edges.(是否要扩大边缘(对边缘进行膨胀操作),默认为True)
        RGB: Indicates if the image format is RGB (True) or BGR (False).
    Returns:
        An edge mask of the input image.
    """
    # Convert the image to grayscale as Canny edge detection requires a single channel image
    # 将输入的RGB图像转换为灰度图像。如果输入图像是RGB格式，则使用 cv2.COLOR_RGB2GRAY 进行转换；如果是BGR格式，则使用 cv2.COLOR_BGR2GRAY 进行转换。
    gray_image = cv2.cvtColor(
        rgb_image, cv2.COLOR_BGR2GRAY if not RGB else cv2.COLOR_RGB2GRAY)
    
    # 检查灰度图像的数据类型，如果不是 np.uint8 类型，则将其转换为 np.uint8 类型。
    if gray_image.dtype != np.uint8:
        gray_image = gray_image.astype(np.uint8)
    
    # 利用Canny边缘检测算法计算图像的边缘
    # threshold1 和 threshold2 是两个阈值参数，用于控制边缘的检测灵敏度；apertureSize 是 Sobel 滤波器的尺寸；L2gradient 表示使用更精确的L2范数计算梯度。
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200, apertureSize=3, L2gradient=True)

    # Define the structuring element for dilation, you can change the size for a thicker/thinner mask
    if dilate:#如果 dilate 为 True，则对边缘进行膨胀操作
        # 创建一个2x2的矩形结构元素 kernel 用于膨胀操作。
        kernel = np.ones((2, 2), np.uint8)
        # 调用 cv2.dilate() 函数对边缘图像进行膨胀操作，以加粗边缘。
        edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def calc_psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """ Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    Args:
        img1: The first image.
        img2: The second image.
    Returns:
        The PSNR value.
    """
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def create_point_cloud(image: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    Creates a point cloud from an image, depth map, camera intrinsics, and pose.
    (从图像,深度图,内参以及pose中获取点云)

    Args:
        image: The RGB image of shape (H, W, 3)
        depth: The depth map of shape (H, W)
        intrinsics: The camera intrinsic parameters of shape (3, 3)
        pose: The camera pose of shape (4, 4)
    Returns:
        A point cloud of shape (N, 6) with last dimension representing (x, y, z, r, g, b)
    """
    # 获取深度图的高度和宽度
    height, width = depth.shape

    # Create a mesh grid of pixel coordinates
    # 生成一个网格坐标系(为整数值)，其中 u 和 v 分别表示图像的宽度和高度
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Convert pixel coordinates to camera coordinates
    # 将像素坐标转换为相机坐标(根据相机内参和深度图)
    x = (u - intrinsics[0, 2]) * depth / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * depth / intrinsics[1, 1]
    z = depth

    # Stack the coordinates together
    # 将坐标沿着最后一个轴（也就是沿着列）进行堆叠。
    # points 将会是一个二维数组，其中每一行代表一个点的坐标。每一行包含了四个元素，分别代表了该点的 x 坐标、y 坐标、z 坐标以及齐次坐标表示中的 1。
    points = np.stack((x, y, z, np.ones_like(z)), axis=-1)

    # Reshape the coordinates for matrix multiplication
    # 将points数组重新塑造成一个二维数组，其中每行包含4个元素。
    # 参数-1的作用是告诉reshape函数根据数组的总元素数量自动计算该轴的长度，这样可以保证数组的总元素数量不变。在这个例子中，由于每个点的坐标由四个元素表示（x、y、z、齐次坐标1），因此将数组重新塑造为每行包含4个元素的形状。这样做的目的可能是为了方便对坐标进行处理或者是与其他格式相匹配。
    points = points.reshape(-1, 4)

    # Transform points to world coordinates
    # 将点从相机坐标系转换到世界坐标系
    posed_points = pose @ points.T
    
    # 对变换后的坐标进行切片操作。
    # 使用 [:, :3] 对数组进行切片操作，保留每个点的前三个元素，即去掉齐次坐标中的最后一个元素。
    # 这样得到的 posed_points 数组就是每个点经过变换后的三维坐标。
    posed_points = posed_points.T[:, :3]

    # Flatten the image to get colors for each point
    # 将图像展平为一个一维数组，其中每个元素代表一个像素的颜色。
    colors = image.reshape(-1, 3)

    # Concatenate posed points with their corresponding color
    # 将变换后的点坐标和颜色进行拼接，得到一个点云。
    point_cloud = np.concatenate((posed_points, colors), axis=-1)

    return point_cloud
