from typing import Annotated, Literal
import numpy as np
import matplotlib.pyplot as plt


Point_3D = Annotated[np.typing.NDArray[np.float32], Literal[3]]

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

def quadratic(A,B,C):
    discriminant :np.float32= np.power(B,2) - (4*A*C)
    print(discriminant)
    if discriminant < 0:
        #returning none because there is no intersection
        return None
    
    discriminant = np.sqrt(discriminant)
    t1 = ((-1*B)+discriminant) / (2*A) #getting solutions
    t2 = ((-1*B)-discriminant) / (2*A)

    return min(t1,t2) #returning the closest intersection



def findIntersection(camerapos, ray,spheres):
    sphere_index: int | None = None
    sphere_intersection: int | None = None
    a: np.float32 = np.dot(ray, ray)
    for i in range(len(spheres)):
        start_to_center: Point_3D = camerapos - spheres[i].center
        b: np.float32 = np.dot(start_to_center, ray)
        b += b
        c: np.float32 = np.dot(start_to_center, start_to_center) - np.square(
            spheres[i].radius
        )

        intersection: np.float32 = quadratic(a, b, c)
        if intersection is not None:
            if sphere_index is not None:
                if intersection < sphere_intersection:
                    sphere_index = i
                    sphere_intersection = intersection
            else:
                sphere_index = i
                sphere_intersection = intersection

    return sphere_index

def createray(e,position):
    x = position[0] - e[0]
    y = position[1] - e[1]
    z = e[2] - position[2]
    return np.array([x,y,z])

''' TODO: Render the scene '''
def render_scene(height,width, plane_height, plane_width,aratio, e,d,spheres,lightpos,lightI,lightC,pixel_colors):
    
    #getting image and plane factors
    positions = np.array([0,0,d+e[2]])
    y = height/plane_height
    x = width/plane_width

    for j in range(len(pixel_colors)):
        for i in range(len(pixel_colors)):
            positions[0] = i/x -1
            positions[1] = j/y -1
            ray = createray(e,positions)
            print(ray)
            closestIntersection = findIntersection(e,ray,spheres)
            print(closestIntersection)
            if closestIntersection is not None:
                pixel_colors[j,i] = spheres[closestIntersection].color

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
    image = render_scene(image_height,image_width,image_plane_height,image_plane_width,aspect_ratio,camera_position,image_plane_dist,spheres,light_position,light_intensity,light_color,pixel_colors)

    plt.imshow(image)
    plt.show()
