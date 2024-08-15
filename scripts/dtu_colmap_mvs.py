import argparse
import numpy as np
import cv2
import os
import collections
import struct
import subprocess
from plyfile import PlyData
import time
import sqlite3

# colmap functions from neus codebase

Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)

    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


# From neus codebase
def load_K_Rt_from_P(P=None):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


# function from colmap for converting rotation matrix to quaternion vector
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


# for doing the inverse, converting quaternion to rotation matrix
def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def run_colmap_mvs(basedir, n_views):
    logfile_name = os.path.join(basedir, "mvs_colmap_" + str(n_views) + "v_output.txt")
    logfile = open(logfile_name, "w")
    workspace = os.path.join(basedir, "dense_" + str(n_views) + "v")
    if not os.path.exists(workspace):
        os.makedirs(workspace)

    # Undistort images
    image_undistorter_args = [
        "colmap",
        "image_undistorter",
        "--image_path",
        os.path.join(basedir, "images_" + str(n_views) + "v"),
        "--input_path",
        os.path.join(basedir, "sparse_" + str(n_views) + "v"),
        "--output_path",
        workspace,
    ]
    if not os.path.exists(os.path.join(workspace, "images")):
        undistorter_output = subprocess.check_output(
            image_undistorter_args, universal_newlines=True
        )
        logfile.write(undistorter_output)
        print("Images undistorted")
    else:
        print("Skipped Images undistorted because they already exist")

    # Apply stereo patch matching (will take some time and needs colmap to be compiled with cuda)
    match_stereo_args = [
        "colmap",
        "patch_match_stereo",
        "--workspace_path",
        workspace,
    ]
    match_stereo_output = subprocess.check_output(
        match_stereo_args, universal_newlines=True
    )
    logfile.write(match_stereo_output)
    print("Stereo Patch Matching done")

    # Fuse the result into a ply file
    fusion_args = [
        "colmap",
        "stereo_fusion",
        "--workspace_path",
        workspace,
        "--output_path",
        os.path.join(workspace, "fused_" + str(n_views) + "v.ply"),
    ]
    fusion_output = subprocess.check_output(fusion_args, universal_newlines=True)
    logfile.write(fusion_output)
    print("Stereo Fusion done")
    ply_data = PlyData.read(os.path.join(workspace, "fused_" + str(n_views) + "v.ply"))
    logfile.close()
    print("Number of points in the mvs point cloud is:", len(ply_data["vertex"]))
    print("Finished running COLMAP MVS, see {} for logs".format(logfile_name))


def create_colmap_db(basedir, n_views):
    logfile_name = os.path.join(basedir, "database_colmap_" + str(n_views) + "v_output.txt")
    logfile = open(logfile_name, "w")

    feature_extractor_args = [
        "colmap",
        "feature_extractor",
        "--database_path",
        os.path.join(basedir, "database_" + str(n_views) + "v.db"),
        "--image_path",
        os.path.join(basedir, "images_" + str(n_views) + "v"),
    ]
    feat_output = subprocess.check_output(
        feature_extractor_args, universal_newlines=True
    )
    logfile.write(feat_output)
    print("Features extracted")

    exhaustive_matcher_args = [
        "colmap",
        "exhaustive_matcher",
        "--database_path",
        os.path.join(basedir, "database_" + str(n_views) + "v.db"),
    ]

    match_output = subprocess.check_output(
        exhaustive_matcher_args, universal_newlines=True
    )
    logfile.write(match_output)
    print("Features matched")
    logfile.close()



def run_colmap_sfm(basedir, n_views):
    logfile_name = os.path.join(basedir, "sfm_colmap_" + str(n_views) + "v_output.txt")
    logfile = open(logfile_name, "w")

    p = os.path.join(basedir, "sparse_" + str(n_views) + "v")
    if not os.path.exists(p):
        os.makedirs(p)

    triangulator_args = [
        "colmap",
        "point_triangulator",
        "--database_path",
        os.path.join(basedir, "database_" + str(n_views) + "v.db"),
        "--image_path",
        os.path.join(basedir, "images_" + str(n_views) + "v"),
        "--input_path",
        os.path.join(basedir, "sfm_intial"),
        "--output_path",
        p,  # --export_path changed to --output_path in colmap 3.6
        "--Mapper.num_threads",
        "16",
        "--Mapper.init_min_tri_angle",
        "4",
        "--Mapper.multiple_models",
        "0",
        "--Mapper.extract_colors",
        "0",
        "--Mapper.tri_ignore_two_view_tracks",
        "0",
    ]

    map_output = subprocess.check_output(triangulator_args, universal_newlines=True)
    logfile.write(map_output)
    logfile.close()
    print("Sparse model created")
    points3dfile = os.path.join(p, "points3D.bin")
    pts3d_sparse = read_points3d_binary(points3dfile)
    print("number of points in sparse point cloud is:", len(pts3d_sparse))
    print("Finished running COLMAP sfm, see {} for logs".format(logfile_name))

# Define a function to convert rows to dictionaries
def rows_to_dict(rows):
    columns = [desc[0] for desc in cursor.description]  # Get column names
    result = []
    for row in rows:
        # Create a dictionary where column names are keys and row values are values
        row_dict = dict(zip(columns, row))
        result.append(row_dict)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_dir", type=str, help="input scene directory")
    parser.add_argument(
        "--n_views", type=int, default=3, help="Number of views used for SfM and MVS"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["novel_view_synthesis", "surface_reconstruction"],
        help="Choose between novel view synthesis or surface reconstruction",
    )
    args = parser.parse_args()

    path_to_cam = os.path.join(args.scan_dir, "cameras.npz")  # cameras_sphere.npz
    path_to_images = os.path.join(args.scan_dir, "image")
    path_to_output = os.path.join(args.scan_dir, "sfm_intial")
    views_n = args.n_views
    images_txt_fname = os.path.join(path_to_output, "images.txt")
    cameras_txt_fname = os.path.join(path_to_output, "cameras.txt")
    points_txt_fname = os.path.join(path_to_output, "points3D.txt")
    os.makedirs(path_to_output, exist_ok=True)

    # load all cameras
    cams = np.load(path_to_cam)
    # loop over all images and write to files
    n_images = max([int(k.split("_")[-1]) for k in cams.keys()]) + 1
    # for sparse novel view synthesis use ids instead of all images
    if args.task == "novel_view_synthesis":
        train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
    else:
        train_idx = []
        for e in os.listdir(path_to_images):
            img_id = int(e.split(".")[0])
            train_idx.append(img_id)

    if views_n > 0:
        indices = train_idx[:views_n]
    else:
        indices = train_idx
    indices.sort()
    n_images = len(indices)

    sparse_images_path = os.path.join(
        args.scan_dir, "images_" + str(n_images) + "v"
    )
    if not os.path.exists(sparse_images_path):
        os.makedirs(sparse_images_path)

    for i in indices:
        img_filename = f"{i:06d}.png"
        img_path = os.path.join(path_to_images, img_filename)
        copy_path = os.path.join(sparse_images_path, img_filename)
        # Copy image to folder containing sparse views
        copy_args = ["cp", img_path, copy_path]
        copy_output = subprocess.check_output(copy_args, universal_newlines=True)

    create_colmap_db(args.scan_dir, n_images)
    conn = sqlite3.connect(os.path.join(args.scan_dir, "database_" + str(n_images) + "v.db"))
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM images')
    rows = cursor.fetchall()
    # Convert rows to dictionaries
    result_dicts = rows_to_dict(rows)
    cursor.close()
    conn.close()
    # Update indices to match colmap's database
    # write to colmap extrinsincs
    # line_id_c = 1
    with open(images_txt_fname, "w") as fid:
        for e in result_dicts:
            img_idx = int(e["name"].split(".")[0])
            img_filename = f"{img_idx:06d}.png"
            # get world matrix
            world_mat, scale_mat = (
                cams[f"world_mat_{img_idx}"],
                cams[f"scale_mat_{img_idx}"],
            )
            P = (world_mat @ scale_mat)[
                :3, :4
            ]  # use this so that the object is centered
            # P = (world_mat)[:3, :4] # use this for ground truth pcl comparison
            K, c2w = load_K_Rt_from_P(P)
            # c2w = np.concatenate([c2w[:, 1:2], -c2w[:,0:1], c2w[:,2:3], c2w[:,3:4]], 1)
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            #
            # convert between dtu and colmap convention
            # in neus code #https://github.com/Totoro97/NeuS/blob/main/preprocess_custom_data/colmap_preprocess/pose_utils.py#LL51C5-L51C65
            # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
            # therefore we need to go from [r, -u, t] to [-u, r, -t]
            # flatten rotation
            Rt_flat = R.flatten()
            qvec = rotmat2qvec(Rt_flat)
            qvec = np.array(tuple(qvec))
            tvec = np.array(tuple(np.squeeze(t)))
            image_header = [e["image_id"], *qvec, *tvec, e["camera_id"], img_filename]
            # line_id_c += 1
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")
            points_strings = []
            fid.write(" ".join(points_strings) + "\n")
    # Writing to colmap intrinsics
    # line_id_c = 1
    model_used = "PINHOLE"  # fx, fy, cx, cy
    with open(cameras_txt_fname, "w") as fid:
        HEADER = (
            "# Camera list with one line of data per camera:\n"
            + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
            + "# Number of cameras: {}\n".format((n_images))
        )
        fid.write(HEADER)
        for e in result_dicts:
            img_idx = int(e["name"].split(".")[0])
            img_filename = f"{img_idx:06d}.png"
            img_path = os.path.join(path_to_images, img_filename)
            # copy_path = os.path.join(sparse_images_path, img_filename)
            # # Copy image to folder containing sparse views
            # copy_args = ["cp", img_path, copy_path]
            # copy_output = subprocess.check_output(copy_args, universal_newlines=True)
            img_file = cv2.imread(img_path)
            H, W = img_file.shape[0], img_file.shape[1]
            # get world matrix
            world_mat, scale_mat = (
                cams[f"world_mat_{i}"],
                cams[f"scale_mat_{i}"],
            )
            P = (world_mat @ scale_mat)[:3, :4]
            # P = (world_mat)[:3, :4]
            K, _ = load_K_Rt_from_P(P)
            fx, fy, cx, cy = (
                K[0, 0],
                K[1, 1],
                K[0, 2],
                K[1, 2],
            )
            to_write = [e["camera_id"], model_used, W, H, fx, fy, cx, cy]
            # line_id_c += 1
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")
    # create empty 3d point cloud file
    with open(points_txt_fname, "w") as fid:
        print("Empty 3D point cloud file created")

    # running colmap for sfm with initialization
    start_time = time.time()
    run_colmap_sfm(args.scan_dir, args.n_views)
    sfm_end_time = time.time()
    print("SFM took {} seconds".format(sfm_end_time - start_time))
    # running colmap for mvs with sfm result
    run_colmap_mvs(args.scan_dir, args.n_views)
    mvs_end_time = time.time()
    print("MVS took {} seconds".format(mvs_end_time - sfm_end_time))
