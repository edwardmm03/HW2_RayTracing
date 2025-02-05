import numpy as np
import matplotlib.pyplot as plt
import math

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

def checkIntersection(camerapos, ray_direction,center,radius):
    L = np.subtract(camerapos, center)
    A = np.dot(ray_direction,ray_direction)
    B = 2 * np.dot(ray_direction,L)
    C = np.dot(L,L) - radius**2
    discriminant = B**2 - 4*A*C
    print(discriminant)

    if discriminant < 0:
        return False
    
    t = (-B - math.sqrt(discriminant))/(2*A)
    return t

''' TODO: Render the scene '''
def render_scene(pixel_colors, e, d, image_width, image_height,spheres,plane_width, plane_height,aspect_ratio):
    
    testsphere = spheres[0]
    c = testsphere.center
    r = testsphere.radius

    l=0
    r= plane_width
    t=0
    b = plane_height

    w = np.array([0,0,1])
    u = np.array([0,1,0])
    v = np.array([1,0,0])

    for x in range(image_width):
        for y in range(image_height):
            pixel_pos_x = l + (((r-l)*(x-.5))/image_width)
            pixel_pos_y = b + (((t-b)*(y +.5))/image_height)
            ray_direction = np.add(-d*w,pixel_pos_x*u, pixel_pos_y*v)
            ray_direction = np.subtract(ray_direction,e)
            direction = ray_direction/np.linalg.norm(ray_direction)
            print(direction)
            t = checkIntersection(e,direction,c,r)
            print(t)
            if t and t >0:
                pixel_colors[y][x] = 255

    return pixel_colors

            


    


# Main function
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


    ''' TODO:
    1. complete the function render_scene() to output the final image
    2. inside the render_scene() function, you need to implement:
        i)   Compute the ray direction
        ii)  Find the closest intersection
        iii) Shade each pixel using Blinn-Phong Shading
    
    '''
    image = render_scene(pixel_colors,camera_position,image_plane_dist,image_width,image_height,spheres,image_plane_width,image_plane_height,aspect_ratio)

    # Display the image
    plt.imshow(image)
    # plt.axis('off')
    plt.show()
