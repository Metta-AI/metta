"""
This file contains a set of functions that are used to generate different types of noise sampling patterns.
Those functions are basically vector transformations (x,y) -> (xi,yi)
They take a point (x,y) corresponding to a pixel in the end terrain and return a new point (xi,yi) to -
sample the noise for this pixel based on the function's logic.

"width" and "height" are used to normalize the coordinates to the range [0,1]
It usually needed to understand where the center of the image is, but it is not used in all functions.
The rest of the parameters are used to widen the range of the function's behavior.

Functions are designed to take in pixel coordinates (x,y) in the form of ranges [0,width) and [0,height) respectively.

Example:
let's say we have a function that takes a point (x,y) and returns (x**2, y**2)
we can add few parameters to make it more interesting:
xi = (ax*x+bx)**2
yi = (ay*y+by)**2
now ax,bx,ay,by are parameters that will allow to create more complex patterns

We can notice that it will generate elliptical patterns, but they will always be aligned with the axes.
To mitigate this we can first add a rotation around the center of the image:

angle_theta = 2*math.pi*t
theta_cos = math.cos(angle_theta)
theta_sin = math.sin(angle_theta)
xi, yi = theta_cos*(x-0.5*width) + theta_sin*(y-0.5*height), -theta_sin*(x-0.5*width) + theta_cos*(y-0.5*height)

xi = (ax*xi+bx)**2
yi = (ay*yi+by)**2
this way we can rotate the pattern around the center of the image.

Sometimes it's beneficial to add excessive parameters
like the general scaling, even though it's technically could be done with the parameters ax,bx,ay,by
This way it's easier to control the high level function's behavior and change the pattern.

xi, yi = xi*zoom, yi*zoom

In the end we can have a function that generates a wide variety of patterns controlled by a lot of parameters:
ax,bx,ay,by,angle_theta,zoom etc
Sampling those parameters randomly gives a lot of variety to the end result.

The parameter's default values usually are set to the most "neutral" values.
Running the function with default values will give a pattern that is not too distorted
and can be used as a base for further modifications or understanding the general behavior.
"""

import math

import numpy as np


def xy_noise(
    x: int,
    y: int,
    width: int,
    height: int,
    x_zoom: float = 0.1,
    y_zoom: float = 0.1,
) -> tuple[float, float]:
    # simple function to generate additional noise, scaled simply along x and y axes

    return (x * x_zoom, y * y_zoom)


def squeezed_noise(
    x: int,
    y: int,
    width: int,
    height: int,
    zoom: float = 0.1,
    squeeze: float = 1.5,
    angle_theta: float = 0.25,
) -> tuple[float, float]:
    # function used in "noise" generator
    # generates additional noise, but scaled globally, tilted arbitrary at angle theta and squeezed
    shift_from_center_for_x = 0
    shift_from_center_for_y = 0
    alpha = 2 * math.pi * angle_theta
    cs = math.cos(alpha)
    sn = math.sin(alpha)

    xi = (+cs * (x - (0.5 + shift_from_center_for_x) * width) + sn * (y - (0.5 + shift_from_center_for_y) * height)) + (
        0.5 + shift_from_center_for_x
    ) * width
    yi = (-sn * (x - (0.5 + shift_from_center_for_x) * width) + cs * (y - (0.5 + shift_from_center_for_y) * height)) + (
        0.5 + shift_from_center_for_y
    ) * height

    xi = (xi - 0.5 * width) * zoom**2 / squeeze
    yi = (yi - 0.5 * height) * zoom**2 * squeeze

    return (xi, yi)


def spiral(
    x: int,
    y: int,
    width: int,
    height: int,
    zoom: float = 0.1,  # global scaling
    squeeze: float = 1.5,  # how squeezed the result
    angle_theta: float = 0.25,  # angle of rotation in 2*pi radians
    P: float = 2.0,  # thickness and direction of the spiral
    xc: float = 0.0,  # x off-center
    yc: float = 0.0,  # y off-center
) -> tuple[float, float]:
    # function used in "spiral" generator
    alpha = 2 * math.pi * angle_theta
    cs = math.cos(alpha)
    sn = math.sin(alpha)
    xi, yi = (
        ((cs * (x - (0.5 + xc) * width) + sn * (y - (0.5 + yc) * height)) + (0.5 + xc) * width),
        ((-sn * (x - (0.5 + xc) * width) + cs * (y - (0.5 + yc) * height)) + (0.5 + yc) * height),
    )

    xi = (xi - (0.5 + xc) * width) * zoom / squeeze
    yi = (yi - (0.5 + yc) * height) * zoom * squeeze
    distance = math.sqrt(xi**2 + yi**2)
    a = distance * P  # the angle of rotation is proportional to the distance from center and P
    xi, yi = xi * math.cos(a) - yi * math.sin(a), yi * math.cos(a) + xi * math.sin(a)

    return (xi, yi)


def arbitrary_tilted_lattice(
    x: int,
    y: int,
    width: int,
    height: int,
    x_zoom: float = 1.5,
    y_zoom: float = 1.5,
    angle_theta: float = 0.0,
    line1_wavelength: int = 3,
    line2_wavelength: int = 3,
    line1_thickness: int = 1,
    line2_thickness: int = 1,
) -> tuple[float, float]:
    # function used in "arbitrary_tilted_lattice" generator
    alpha = 2 * math.pi * angle_theta
    alpha_tangent = math.tan(alpha)
    line1 = math.floor(x + alpha_tangent * y)
    line2 = math.floor(alpha_tangent * x - y)
    if line1 % line1_wavelength < line1_thickness or line2 % line2_wavelength < line2_thickness:
        if line1 % line1_wavelength < line1_thickness:
            xi, yi = x_zoom * line1, y_zoom * (line2 - line2 % line2_wavelength)
        else:
            xi, yi = x_zoom * (line1 - line1 * line1_wavelength), y_zoom * line2
    else:
        xi, yi = 0, 0

    return (xi, yi)


def arbitrary_tilted_napkin(
    x: int,
    y: int,
    width: int,
    height: int,
    x_zoom: float = 1.5,
    y_zoom: float = 1.5,
    angle_theta: float = 0.0,
    line1_wavelength: int = 3,
    line2_wavelength: int = 3,
    line1_thickness: int = 1,
    line2_thickness: int = 1,
) -> tuple[float, float]:
    # function used in "arbitrary_tilted_napkin" generator
    # During my attempts to modify "the lattice" function I've made a few mistakes
    # As a result, it produced an interesting pattern. I don't know why, but it works
    alpha = 2 * math.pi * angle_theta
    alpha_tangent = math.tan(alpha)
    line1 = math.floor(x + alpha_tangent * y)
    line2 = math.floor(alpha_tangent * x - y)
    if line1 % line1_wavelength < line1_thickness or line2 % line2_wavelength < line2_thickness:
        if line1 % line1_wavelength < line1_thickness:
            xi, yi = (
                x_zoom * line1,
                y_zoom * (line1 - line1 % line1_wavelength),
            )  # line2 arguments were swapped for line1 by a mistake
        else:
            xi, yi = (
                x_zoom * (line2 - line2 * line2_wavelength),
                y_zoom * line2,
            )  # line1 arguments were swapped for line2 by a mistake
    else:
        xi, yi = 0, 0

    return (xi, yi)


def the_sphere(
    x: int,
    y: int,
    width: int,
    height: int,
    x_zoom: float = 0.1,
    y_zoom: float = 0.1,
    angle_theta: float = 0.25,
    x_pow: int = 2,
    y_pow: int = 2,
    xc: float = 0.0,
    yc: float = 0.0,
    P: float = 1.0,
    ax: float = 0.0,
    ay: float = 0.0,
    bx: float = 0.0,
    by: float = 0.0,
) -> tuple[float, float]:
    # function used in "the_sphere" and "the_what" generator
    # I don't know how to easily describe what this function does internally on a high level
    # it's several complicated functions used almost randomly with a lot of excess parameters to produce high variety
    # But it generates interesting shapes in the end
    # Just like the previous "napkin" function, this one was made by accident during "symmetry" development
    xi = (x - (0.5 + xc) * width) * 0.05
    yi = (y - (0.5 + yc) * height) * 0.05
    alpha = 2 * math.pi * angle_theta
    cs = math.cos(alpha)
    sn = math.sin(alpha)
    xi, yi = (cs * (xi) + sn * (yi)), (-sn * (xi) + cs * (yi))
    a = np.sinc(
        ((bx + xi) ** x_pow + (by + yi) ** y_pow)
        * math.sin(math.atan2((y - (0.5 + ay) * height), (x - (0.5 + ax) * width)) * P)
    )
    xi, yi = float(a), float(a)

    return (xi * x_zoom, yi * y_zoom)


def cross_curse(
    x: int,
    y: int,
    width: int,
    height: int,
    x_zoom: float = 0.1,
    y_zoom: float = 0.1,
    angle_theta: float = 0.25,
    x_pow: int = 2,
    y_pow: int = 2,
    xc: float = 0.0,
    yc: float = 0.0,
) -> tuple[float, float]:
    # function used in "cross_curse" generator
    alpha = 2 * math.pi * angle_theta
    cs = math.cos(alpha)
    sn = math.sin(alpha)
    xi, yi = x, y

    xi, yi = (
        (cs * (xi - (0.5 + xc) * width) + sn * (yi - (0.5 + yc) * height)),
        (-sn * (xi - (0.5 + xc) * width) + cs * (yi - (0.5 + yc) * height)),
    )

    xi, yi = 8 * xi**x_pow / x_pow**x_pow, 8 * yi**y_pow / y_pow**y_pow

    return (xi * x_zoom, yi * y_zoom)


def radial_symmetry(
    x: int,
    y: int,
    width: int,
    height: int,
    x_zoom: float = 0.1,
    y_zoom: float = 0.1,
    angle_theta: float = 0.25,
    symmetry: int = 3,
    xc: float = 0.0,
    yc: float = 0.0,
) -> tuple[float, float]:
    # function used in "symmetry" generator
    alpha = 2 * math.pi * angle_theta
    cs = math.cos(alpha)
    sn = math.sin(alpha)
    xi, yi = (
        (cs * (x - 0.5 * width + xc * width) + sn * (y - 0.5 * height + yc * height)),
        (-sn * (x - 0.5 * width + xc * width) + cs * (y - 0.5 * height + yc * height)),
    )

    beta = (symmetry - 1) * math.atan2((yi), (xi))  # I don't know why it works properly with (symmetry-1)
    csb = math.cos(beta)
    snb = math.sin(beta)
    xi, yi = (csb * xi - snb * yi), (snb * xi + csb * yi)

    xi, yi = (
        (cs * (xi - 0.5 * width + xc * width) - sn * (yi - 0.5 * height + yc * height)),
        (sn * (xi - 0.5 * width + xc * width) + cs * (yi - 0.5 * height + yc * height)),
    )

    return (xi * x_zoom, yi * y_zoom)
