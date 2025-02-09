import numpy as np
import matplotlib.pyplot as plt


# Define the scene
class Sphere:
    def __init__(self, center, radius, color, ka, kd, ks, shininess):
        self.center = np.array(center)  # The 3D coordinates of the sphere's center
        self.radius = radius    # The radius of the sphere
        self.color = np.array(color)    # The RGB color of the sphere
        self.ka = ka  # Ambient coefficient
        self.kd = kd  # Diffuse coefficient
        self.ks = ks  # Specular coefficient
        self.shininess = shininess  # he shininess factor for specular highlights


def intersection(e,ray,spheres):
    #returns the closest sphere that the ray intersects
    
    #initializing to none in case there is no intersection
    sphere_index = None
    sphere_intersection = None
    
    A = np.dot(ray, ray) #creating A for quadratic formula
    
    for i in range(len(spheres)):
        start_to_center =e- spheres[i].center
        B = np.dot(start_to_center, ray)
        B += B
        C = np.dot(start_to_center, start_to_center) - np.square(spheres[i].radius)

        discriminant = np.power(B, 2) - (4 * A * C)

        if discriminant < 0:
            #the ray does not intersect the sphere
            intersection = None
        else:
            discriminant = np.sqrt(discriminant)

            t1 = ((-1 * B) + discriminant) / (2 * A)
            t2 = ((-1 * B) - discriminant) / (2 * A)
            intersection = min(t1,t2) #returning the closest intersection point

        if intersection is not None:
            #if there is an intersection
            if sphere_index is not None:
                #if one sphere is already intersected set the sphere the ray intersects to the sphere closer to the camera
                if intersection < sphere_intersection:
                    sphere_index = i
                    sphere_intersection = intersection
            else:
                sphere_index = i
                sphere_intersection = intersection

    return sphere_index


def shade(sphere):
    return sphere.color


def render_scene(image_height,image_width,plane_height,plane_width,pixel_colors,e,d,spheres):
    """TODO:
    1. complete the function render_scene() to output the final image
    2. inside the render_scene() function, you need to implement:
        i)   Compute the ray direction
        ii)  Find the closest intersection
        iii) Shade each pixel using Blinn-Phong Shading
    """
    #relating pixels from image to imapge plane
    pixel_position = np.array(
        [0, 0, d + e[2]],
        dtype=np.float32,
    )
    y = image_height / plane_height
    x = image_width / plane_width

    #going through each pixel in the image to calculate the ray, intersection, and shading color
    for j in range(len(pixel_colors[1])):
        for i in range(len(pixel_colors)):
            pixel_position[0] = i / x - 1
            pixel_position[1] = j / y - 1
            ray = np.array([pixel_position[0] - e[0],pixel_position[1] - e[1], e[2] - pixel_position[2]])
            print(ray)

            close_sphere: int = intersection(e, ray, spheres) #get the closest sphere that is intersected
            if close_sphere is not None:
                pixel_colors[j, i] = shade(spheres[close_sphere])

    return pixel_colors


if __name__ == "__main__":
# Define spheres
    spheres = [
        Sphere(center=[0, 0, -5], radius=1, color=[0, 1, 1], ka=0.1, kd=0.7, ks=0.5, shininess=32),  # Cyan
        Sphere(center=[2, 0, -6], radius=1.5, color=[1, 0, 1], ka=0.1, kd=0.7, ks=0.5, shininess=32),  # Magenta
    ]

    # Define light source
    light_position = np.array([5, 5, -10])
    light_intensity = 1.0
    light_color = np.array([1, 1, 1])  # White light

    # Render the scene
    image_width = 800
    image_height = 800

    # Initial pixel colors of the scene (final output image)
    pixel_colors = np.zeros((image_height, image_width, 3))

    # Define the image plane
    image_plane_height = 2.0
    aspect_ratio = image_width / image_height
    image_plane_width = aspect_ratio * image_plane_height

    # Define the camera
    camera_position = np.array([0, 0, 0])

    # Distance between the camera and the image plane
    image_plane_dist = 1.0

    image= render_scene(image_height,image_width,image_plane_height,image_plane_width,pixel_colors,camera_position,image_plane_dist, spheres)

    # Display the image
    plt.imshow(image)
    # plt.axis('off')
    plt.show()
