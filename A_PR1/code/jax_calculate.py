import jax.numpy as jnp
import jax
import numpy as np
import math

# Quaternion multiplication (JAX version)
def qmult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return jnp.array([w, x, y, z])

# Quaternion exponential map (JAX version)
def qexp(q):
    qs = q[0]  # Extract the real part
    qv = q[1:]  # Extract the imaginary part
    norm_qv = jnp.linalg.norm(qv)

    def small_angle_case(_):
        return jnp.array([jnp.exp(qs), 0.0, 0.0, 0.0])

    def normal_case(_):
        exp_qs = jnp.exp(qs)
        cos_norm_qv = jnp.cos(norm_qv)
        sin_norm_qv = jnp.sin(norm_qv) / norm_qv
        qv_part = sin_norm_qv * qv
        return jnp.concatenate([jnp.array([exp_qs * cos_norm_qv]), exp_qs * qv_part])

    return jax.lax.cond(norm_qv < 1e-8, small_angle_case, normal_case, operand=None)

# Quaternion inverse (JAX version)
def qinverse(q):
    return jnp.array([q[0], -q[1], -q[2], -q[3]]) / (jnp.dot(q, q) + 1e-8)

# Quaternion logarithmic map
def log_quaternion(q):
    """JAX-compatible quaternion logarithmic map"""
    w, x, y, z = q
    norm_v = jnp.sqrt(x**2 + y**2 + z**2)

    def zero_case(_):
        return jnp.array([0.0, 0.0, 0.0, 0.0])

    def normal_case(_):
        theta = 2 * jnp.arccos(w)
        sin_theta_2 = jnp.sin(theta / 2)
        return jnp.array([0, (x / sin_theta_2) * (theta / 2), (y / sin_theta_2) * (theta / 2), (z / sin_theta_2) * (theta / 2)])

    return jax.lax.cond(norm_v < 1e-8, zero_case, normal_case, operand=None)

def safe_log_quaternion(q):
    # Clip the real part w to [-1, 1] to avoid arccos domain error
    w = jnp.clip(q[0], -1.0, 1.0)
    # Extract the imaginary part
    v = q[1:]
    # Compute the norm of the imaginary part
    norm_v = jnp.linalg.norm(v)
    # Compute the rotation angle
    theta = jnp.arccos(w)

    # Handle the case where the norm of the imaginary part is close to 0
    # by directly returning a zero vector
    # Otherwise, compute the logarithmic map according to the formula
    log_v = jnp.where(
        norm_v < 1e-8,
        jnp.zeros_like(v),
        (theta / norm_v) * v
    )

    # The logarithmic map result has a real part of 0 and an imaginary part of the computed log_v
    return jnp.concatenate([jnp.array([0.0]), log_v])

def quat2euler(q):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw).
    
    Parameters:
        q: Quaternion, in the format of (w, x, y, z)
    
    Returns:
        [roll, pitch, yaw] (in radians)
    """
    w, x, y, z = q

    # Compute roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Compute pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if sinp >= 1:
        pitch = math.pi / 2
    elif sinp <= -1:
        pitch = -math.pi / 2
    else:
        pitch = math.asin(sinp)

    # Compute yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return [roll, pitch, yaw]


def normalize_angle(angle):
    """Normalize the angle to be within the range of [-π, π]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def fix_angle_continuity(angle_sequence):
    """
    Expand the angle sequence to avoid discontinuities.
    Note that this will not strictly limit the angles to the range of [-π, π],
    but will prevent large discontinuities.
    """
    angle_sequence = np.asarray(angle_sequence)
    fixed_angles = np.zeros_like(angle_sequence)
    
    # Use the first frame as a reference
    fixed_angles[0] = angle_sequence[0]
    
    for i in range(1, len(angle_sequence)):
        # Compute raw_diff = the difference between the current frame and the previous frame
        raw_diff = angle_sequence[i] - fixed_angles[i - 1]
        # Wrap the difference to be within the range of [-π, π]
        wrapped_diff = (raw_diff + np.pi) % (2 * np.pi) - np.pi
        
        # Add the wrapped difference to the previous frame to get the new frame
        fixed_angles[i] = fixed_angles[i - 1] + wrapped_diff
    
    return fixed_angles

def mat2euler(R_matrices):
    """
    Convert a stack of 3x3 rotation matrices (R) to Euler angles (roll, pitch, yaw),
    assuming the rotation order is 'szyx' (first X, then Y, then Z) in the "static" angle representation.
    
    Parameters:
        R_matrices: A stack of rotation matrices with shape (3, 3, N)
    
    Returns:
        euler_angles: An array of shape (N, 3) containing the Euler angles in radians
    """
    num_samples = R_matrices.shape[2]
    euler_angles = np.zeros((num_samples, 3))

    for i in range(num_samples):
        R = R_matrices[:, :, i]

        # Corresponding to 'szyx' decomposition:
        #   R = Rz(psi) * Ry(theta) * Rx(phi)
        #   Here, we use the naming convention: roll=phi, pitch=theta, yaw=psi
        #
        # But be careful about which "Euler angle order" you actually want.
        # The following is similar to your original code:

        # pitch = asin(-R[2,0])
        sinp = -R[2, 0]
        # Ensure that the result is within [-1, 1]
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = math.asin(sinp)

        cos_pitch = math.cos(pitch)
        if abs(cos_pitch) > 1e-6:
            # roll = atan2(R[2,1], R[2,2])
            roll = math.atan2(R[2, 1], R[2, 2])
            # yaw  = atan2(R[1,0], R[0,0])
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            # pitch ~ ±π/2, a gimbal-like situation occurs
            # You can handle this case according to your needs
            roll = 0.0
            yaw = math.atan2(-R[0, 1], R[1, 1])

        # Store the results directly without limiting each frame to [-π, π]
        euler_angles[i, 0] = roll
        euler_angles[i, 1] = pitch
        euler_angles[i, 2] = yaw

    # ---- Expand each channel on the time axis ----
    # np.unwrap can directly handle 2D arrays with the specified column; axis=0 means expanding on the time axis
    # euler_angles = np.unwrap(euler_angles, axis=0)

    # If you insist on having each frame within [-π, π], you can map it again
    # But this will again cause jumps at some time points. Usually, it's not necessary.
    # euler_angles = normalize_angle_array(euler_angles)

    return euler_angles

def quat2mat(q):
    """
    Convert a quaternion to a rotation matrix
    
    Parameters:
        q: A quaternion in the format (w, x, y, z)
    
    Returns:
        A 3x3 rotation matrix
    """
    w, x, y, z = q
    r00 = 1 - 2 * y**2 - 2 * z**2
    r01 = 2 * x * y - 2 * z * w
    r02 = 2 * x * z + 2 * y * w
    r10 = 2 * x * y + 2 * z * w
    r11 = 1 - 2 * x**2 - 2 * z**2
    r12 = 2 * y * z - 2 * x * w
    r20 = 2 * x * z - 2 * y * w
    r21 = 2 * y * z + 2 * x * w
    r22 = 1 - 2 * x**2 - 2 * y**2
    return np.array([[r00, r01, r02],
                     [r10, r11, r12],
                     [r20, r21, r22]])
