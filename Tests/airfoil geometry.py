import numpy as np

def NACA_mesh(t, p, c):
    # Coordinates of the Test Volume

    x = np.linspace(0, 3, 30)
    y = np.linspace(0, 1, 10)

    # Mesh Coordinates of the Test Volume
    X, Y = np.meshgrid(x, y)

    # Start Point Reference for Airfoil
    START_POINT = np.array([0.5, 0.5])

    # Bool of whether x-coordinate is after the start point
    x_after_start = x > START_POINT[0]

    # Bool of whether x-coordinate is before the end point
    x_before_end = x < (START_POINT[0] + 1)

    # Bool of whether x-coordinate is between the start and end points
    x_airfoil_mask = np.logical_and(x_after_start, x_before_end)

    # x-coordinates which are inside the airfoil
    x_airfoil = x[x_airfoil_mask]

    # x-coordinates inside the airfoil, relative to the start point
    x_rel = x_airfoil - START_POINT[0]

    # Thickness of the airfoil at x-coordinates relative to the the start point
    y_t = 5 * t * (0.2969 * np.sqrt(x_rel) -
                   0.1260 * x_rel -
                   0.3516 * x_rel**2 +
                   0.2843 * x_rel**3 +
                   0.1015 * x_rel**4
                   )

    # Camber (Front Formula) at x-coordinates relative to the the start point
    y_c_front = c/(p**2) * (2 * p * x_rel - x_rel**2)

    # Camber (Back Formula) at x-coordinates relative to the the start point
    y_c_back = c/((1-p)**2) * ((1 - 2 * p) + 2*p*x_rel - x_rel**2)

    # Bool of whether x-coordinate, relative to the start point, is before the point of maximum camber
    front = x_rel < p

    # Camber (Total Formula) at x-coordinates relative to the start point
    y_c = front * y_c_front + (1-front) * y_c_back

    # y-coordinates of the top of the airfoil
    y_top = y_t + y_c + START_POINT[1]

    #y-coordinates of the bottom of the airfoil
    y_bottom = y_t - y_c + START_POINT[1]

    _, Y_TOP = np.meshgrid(x, y_top)
    _, Y_BOTTOM = np.meshgrid(x, y_bottom)

    Y_TOP_MASK = Y < Y_TOP
    Y_BOTTOM_MASK = Y > Y_BOTTOM

    Y_AIRFOIL_MASK = np.logical_and(Y_TOP_MASK, Y_BOTTOM_MASK)

    X_AIRFOIL_MASK, _ = np.meshgrid(x_airfoil_mask, y)

    AIRFOIL_MASK = np.logical_and(X_AIRFOIL_MASK,Y_AIRFOIL_MASK)

    return AIRFOIL_MASK

if __name__ == '__main__':
    NACA_mesh(0.12, 0.40, 0.2)