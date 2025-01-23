import numpy as np


def get_zernike_mode(order, theta, phi):
    """
    Returns phase of Zernike mode order = (n,m) for pupil angles (theta, phi).

    Parameters
    ----------
    order
    theta
    phi

    Returns
    -------
    zernikemode
    """

    n, m = order
    m_abs = np.abs(m)
    rho = theta / theta.max()
    zernikemode = np.zeros(theta.shape)
    for k in range(int((n - m_abs) / 2 + 1)):
        zernikemode += int((-1) ** k * np.math.factorial(n - k)
                           / (np.math.factorial(k) * np.math.factorial(int((n + m_abs) / 2 - k))
                              * np.math.factorial(int((n - m_abs) / 2 - k)))) \
                       * rho ** (n - 2 * k)
    if m >= 0:
        zernikemode *= np.cos(m * phi)
    else:
        zernikemode *= -np.sin(m * phi)

    zernikemode *= np.sqrt((2 - (m == 0)) * (n + 1))

    return zernikemode
