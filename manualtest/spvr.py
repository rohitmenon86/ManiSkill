import numpy as np
import sapien
from sapien import Entity, Scene
from sapien import internal_renderer as R
from sapien.render import (
    RenderSystem,
    RenderVRDisplay,
    RenderWindow,
    SapienRenderer,
    get_viewer_shader_dir,
)

sapien.render.set_log_level("info")
sapien.render.set_viewer_shader_dir("../vulkan_shader/vr_default")


class VRViewer:
    def __init__(self):
        self.vr = RenderVRDisplay()
        self.controllers = self.vr.get_controller_ids()
        self.renderer_context = sapien.render.SapienRenderer()._internal_context
        self._create_visual_models()

        self.reset()

    def reset(self):
        self.controller_axes = None
        self.marker_spheres = None

    @property
    def root_pose(self):
        return self.vr.root_pose

    @root_pose.setter
    def root_pose(self, pose):
        self.vr.root_pose = pose

    @property
    def ray_angle(self):
        return np.pi / 4

    @property
    def render_scene(self):
        return self.vr._internal_scene

    @property
    def controller_poses(self):
        return [self.vr.get_controller_pose(c) for c in self.controllers]

    def set_scene(self, scene):
        self.scene = scene
        self.vr.set_scene(scene)

    def render(self):
        self._update_controller_axes()
        self.vr.update_render()
        self.vr.render()

    def pick(self, index):
        t2c = sapien.Pose()
        t2c.rpy = [0, self.ray_angle, 0]
        c2r = self.controller_poses[index]
        r2w = self.root_pose
        t2w = r2w * c2r * t2c
        d = t2w.to_transformation_matrix()[:3, 0]

        assert isinstance(self.scene.physx_system, sapien.physx.PhysxCpuSystem)
        px: sapien.physx.PhysxCpuSystem = self.scene.physx_system
        res = px.raycast(t2w.p, d, 50)
        return res

    # helper visuals
    def _create_visual_models(self):
        self.cone = self.renderer_context.create_cone_mesh(16)
        self.capsule = self.renderer_context.create_capsule_mesh(0.1, 0.5, 16, 4)
        self.cylinder = self.renderer_context.create_cylinder_mesh(16)
        self.sphere = self.renderer_context.create_uvsphere_mesh()

        self.laser = self.renderer_context.create_line_set(
            [0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]
        )

        self.mat_red = self.renderer_context.create_material(
            [5, 0, 0, 1], [0, 0, 0, 1], 0, 1, 0
        )
        self.mat_green = self.renderer_context.create_material(
            [0, 1, 0, 1], [0, 0, 0, 1], 0, 1, 0
        )
        self.mat_blue = self.renderer_context.create_material(
            [0, 0, 1, 1], [0, 0, 0, 1], 0, 1, 0
        )
        self.mat_cyan = self.renderer_context.create_material(
            [0, 1, 1, 1], [0, 0, 0, 1], 0, 1, 0
        )
        self.mat_magenta = self.renderer_context.create_material(
            [1, 0, 1, 1], [0, 0, 0, 1], 0, 1, 0
        )
        self.mat_white = self.renderer_context.create_material(
            [1, 1, 1, 1], [0, 0, 0, 1], 0, 1, 0
        )
        self.red_cone = self.renderer_context.create_model([self.cone], [self.mat_red])
        self.green_cone = self.renderer_context.create_model(
            [self.cone], [self.mat_green]
        )
        self.blue_cone = self.renderer_context.create_model(
            [self.cone], [self.mat_blue]
        )
        self.red_capsule = self.renderer_context.create_model(
            [self.capsule], [self.mat_red]
        )
        self.green_capsule = self.renderer_context.create_model(
            [self.capsule], [self.mat_green]
        )
        self.blue_capsule = self.renderer_context.create_model(
            [self.capsule], [self.mat_blue]
        )
        self.cyan_capsule = self.renderer_context.create_model(
            [self.capsule], [self.mat_cyan]
        )
        self.magenta_capsule = self.renderer_context.create_model(
            [self.capsule], [self.mat_magenta]
        )
        self.white_cylinder = self.renderer_context.create_model(
            [self.cylinder], [self.mat_white]
        )
        self.red_sphere = self.renderer_context.create_model(
            [self.sphere], [self.mat_red]
        )

    def _create_coordiate_axes(self):
        render_scene = self.render_scene

        node = render_scene.add_node()
        obj = render_scene.add_object(self.red_cone, node)
        obj.set_scale([0.5, 0.2, 0.2])
        obj.set_position([1, 0, 0])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.red_capsule, node)
        obj.set_position([0.52, 0, 0])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.green_cone, node)
        obj.set_scale([0.5, 0.2, 0.2])
        obj.set_position([0, 1, 0])
        obj.set_rotation([0.7071068, 0, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.green_capsule, node)
        obj.set_position([0, 0.51, 0])
        obj.set_rotation([0.7071068, 0, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.blue_cone, node)
        obj.set_scale([0.5, 0.2, 0.2])
        obj.set_position([0, 0, 1])
        obj.set_rotation([0, 0.7071068, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.blue_capsule, node)
        obj.set_position([0, 0, 0.5])
        obj.set_rotation([0, 0.7071068, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_line_set(self.laser, node)
        obj.set_scale([40, 0, 0])
        obj.line_width = 20
        ray_pose = sapien.Pose()
        ray_pose.rpy = [0, self.ray_angle, 0]
        obj.set_rotation(ray_pose.q)

        node.set_scale([0.025, 0.025, 0.025])

        return node

    def _update_controller_axes(self):
        if self.controller_axes is None:
            self.controller_axes = [
                self._create_coordiate_axes() for c in self.controllers
            ]

        for n, pose in zip(self.controller_axes, self.controller_poses):
            c2w = self.vr.root_pose * pose
            n.set_position(c2w.p)
            n.set_rotation(c2w.q)

    def _create_marker_sphere(self):
        node = self.render_scene.add_object(self.red_sphere)
        node.set_scale([0.05] * 3)
        node.shading_mode = 0
        node.cast_shadow = False
        node.transparency = 1
        return node

    def update_marker_sphere(self, i, hit):
        if self.marker_spheres is None:
            self.marker_spheres = [
                self._create_marker_sphere() for c in self.controllers
            ]

        if hit is None:
            self.marker_spheres[i].transparency = 1
        else:
            self.marker_spheres[i].set_position(hit.position)
            self.marker_spheres[i].transparency = 0


def run():
    scene = sapien.Scene()
    scene.load_widget_from_package("demo_arena", "DemoArena")

    viewer = VRViewer()
    viewer.set_scene(scene)

    viewer.root_pose = sapien.Pose([1, 0, 0], [0, 0, 0, 1])

    prev_button_pressed = [0 for c in viewer.controllers]
    while True:
        scene.step()
        viewer.render()

        for i, c in enumerate(viewer.controllers):
            button_pressed = viewer.vr.get_controller_button_pressed(c)
            button_pressed ^ prev_button_pressed[i]

            # trigger down
            # if changed & 0x200000000 and button_pressed & 0x200000000:
            #     viewer.pick(i)

            # continuously test
            if button_pressed & 0x200000000:
                hit = viewer.pick(i)
                viewer.update_marker_sphere(i, hit)
            else:
                viewer.update_marker_sphere(i, None)

            prev_button_pressed[i] = button_pressed

            # print(f"{c} button {viewer.vr.get_controller_button_pressed(c):x}")
            # print(f"{c} touch {viewer.vr.get_controller_button_touched(c):x}")
            # print(f"{c} axis 0 {viewer.vr.get_controller_axis_state(c, 0)}")
            # print(f"{c} axis 1 {viewer.vr.get_controller_axis_state(c, 1)}")
            # print(f"{c} axis 2 {viewer.vr.get_controller_axis_state(c, 2)}")


def main():
    try:
        run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()