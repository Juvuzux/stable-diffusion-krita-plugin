from functools import partial
import re
import os
import urllib.parse
import urllib.request
import json

from krita import *

default_url = "http://127.0.0.1:8000"

samplers = ["DDIM", "PLMS", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms']
realesrgan_models = ['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B']


class Script(QObject):
    def __init__(self):
        # Persistent settings (should reload between Krita sessions)
        self.config = QSettings(QSettings.IniFormat, QSettings.UserScope, "krita", "krita_diff_plugin")
        self.restore_defaults(if_empty=True)
        self.working = False

    def cfg(self, name, type):
        return self.config.value(name, type=type)

    def set_cfg(self, name, value, if_empty=False):
        if not if_empty or not self.config.contains(name):
            self.config.setValue(name, value)

    def restore_defaults(self, if_empty=False):
        self.set_cfg('base_url', default_url, if_empty)
        self.set_cfg('just_use_yaml', False, if_empty)
        self.set_cfg('create_mask_layer', True, if_empty)
        self.set_cfg('delete_temp_files', True, if_empty)
        self.set_cfg('workaround_timeout', 100, if_empty)
        self.set_cfg('png_quality', -1, if_empty)

        self.set_cfg('txt2img_prompt', "", if_empty)
        self.set_cfg('txt2img_sampler', samplers.index("k_euler_a"), if_empty)
        self.set_cfg('txt2img_ddim_steps', 20, if_empty)
        self.set_cfg('txt2img_cfg_scale', 7.5, if_empty)
        self.set_cfg('txt2img_batch_count', 1, if_empty)
        self.set_cfg('txt2img_batch_size', 1, if_empty)
        self.set_cfg('txt2img_base_size', 512, if_empty)
        self.set_cfg('txt2img_max_size', 704, if_empty)
        self.set_cfg('txt2img_seed', "", if_empty)

        self.set_cfg('img2img_prompt', "", if_empty)
        self.set_cfg('img2img_sampler', samplers.index("k_euler_a"), if_empty)
        self.set_cfg('img2img_ddim_steps', 50, if_empty)
        self.set_cfg('img2img_cfg_scale', 12.0, if_empty)
        self.set_cfg('img2img_denoising_strength', 0.40, if_empty)
        self.set_cfg('img2img_batch_count', 1, if_empty)
        self.set_cfg('img2img_batch_size', 1, if_empty)
        self.set_cfg('img2img_base_size', 512, if_empty)
        self.set_cfg('img2img_max_size', 704, if_empty)
        self.set_cfg('img2img_seed', "", if_empty)

        self.set_cfg('upscale_realesrgan_model', realesrgan_models.index('RealESRGAN_x4plus'), if_empty)
        self.set_cfg('upscale_base_size', 512, if_empty)
        self.set_cfg('upscale_max_size', 1024, if_empty)

    def update_config(self):
        self.app = Krita.instance()
        self.doc = self.app.activeDocument()
        self.selection = self.doc.selection()

        if self.selection is None:
            self.x = 0
            self.y = 0
            self.width = self.doc.width()
            self.height = self.doc.height()
        else:
            self.x = self.selection.x()
            self.y = self.selection.y()
            self.width = self.selection.width()
            self.height = self.selection.height()

        with urllib.request.urlopen(self.cfg('base_url', str) + '/config') as req:
            res = req.read()
            self.opt = json.loads(res)

    # Server API    @staticmethod
    def post(self, url, body):
        req = urllib.request.Request(url)
        req.add_header('Content-Type', 'application/json')
        body = json.dumps(body)
        body_encoded = body.encode('utf-8')
        req.add_header('Content-Length', str(len(body_encoded)))
        with urllib.request.urlopen(req, body_encoded) as res:
            return json.loads(res.read())

    def txt2img(self):
        params = {
            "orig_width": self.width,
            "orig_height": self.height,
            "prompt": self.cfg('txt2img_prompt', str) if not self.cfg('txt2img_prompt', str).isspace() else None,
            "sampler_name": samplers[self.cfg('txt2img_sampler', int)],
            "ddim_steps": self.cfg('txt2img_ddim_steps', int),
            "cfg_scale": self.cfg('txt2img_cfg_scale', float),
            "batch_count": self.cfg('txt2img_batch_count', int),
            "batch_size": self.cfg('txt2img_batch_size', int),
            "base_size": self.cfg('txt2img_base_size', int),
            "max_size": self.cfg('txt2img_max_size', int),
            "seed": self.cfg('txt2img_seed', str) if not self.cfg('txt2img_seed', str).isspace() else None
        } if not self.cfg('just_use_yaml', bool) else {
            "orig_width": self.width,
            "orig_height": self.height
        }
        return self.post(self.cfg('base_url', str) + '/txt2img', params)

    def img2img(self, path):
        params = {
            "src_path": path,
            "prompt": self.cfg('img2img_prompt', str) if not self.cfg('img2img_prompt', str).isspace() else None,
            "sampler_name": samplers[self.cfg('img2img_sampler', int)],
            "ddim_steps": self.cfg('img2img_ddim_steps', int),
            "cfg_scale": self.cfg('img2img_cfg_scale', float),
            "denoising_strength": self.cfg('img2img_denoising_strength', float),
            "batch_count": self.cfg('img2img_batch_count', int),
            "batch_size": self.cfg('img2img_batch_size', int),
            "base_size": self.cfg('img2img_base_size', int),
            "max_size": self.cfg('img2img_max_size', int),
            "seed": self.cfg('img2img_seed', str) if not self.cfg('img2img_seed', str).isspace() else None
        } if not self.cfg('just_use_yaml', bool) else {
            "src_path": path
        }
        return self.post(self.cfg('base_url', str) + '/img2img', params)

    def upscale(self, path):
        params = {
            "src_path": path,
            "base_size": self.cfg('upscale_base_size', int),
            "max_size": self.cfg('upscale_max_size', int),
            "realesrgan_model": realesrgan_models[self.cfg('upscale_realesrgan_model', int)]
        } if not self.cfg('just_use_yaml', bool) else {
            "src_path": path
        }
        return self.post(self.cfg('base_url', str) + '/upscale', params)

    def save_img(self, path):
        pixel_bytes = self.doc.pixelData(self.x, self.y, self.width, self.height)
        image_data = QImage(pixel_bytes, self.width, self.height, QImage.Format_RGBA8888).rgbSwapped()
        image_data.save(path, "PNG", self.cfg('png_quality', int))
        print(f"Saved image: {path}")

    # Krita tools
    def create_layer(self, name):
        root = self.doc.rootNode()
        layer = self.doc.createNode(name, "paintLayer")
        root.addChildNode(layer, None)
        print(f"created layer: {layer}")
        return layer

    def image_to_ba(self, image):
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        return QByteArray(ptr.asstring())

    def insert_img(self, path, visible=True):
        image = QImage()
        image.load(path, "PNG")
        ba = self.image_to_ba(image)

        layer = self.create_layer(path)
        if not visible:
            layer.setVisible(False)

        layer.setPixelData(ba, self.x, self.y, self.width, self.height)
        print(f"Inserted image: {path}")

    def apply_txt2img(self):
        response = self.txt2img()
        outputs = response['outputs']
        print(f"Getting images: {outputs}")
        for i, output in enumerate(outputs):
            self.insert_img(output, i + 1 == len(outputs))
        self.clear_temp_images(None, outputs)
        self.doc.refreshProjection()

    def apply_img2img(self):
        path = self.opt['new_img']
        self.save_img(path)
        response = self.img2img(path)
        outputs = response['outputs']
        print(f"Getting images: {outputs}")
        for i, output in enumerate(outputs):
            self.insert_img(output, i + 1 == len(outputs))
        self.clear_temp_images(path, outputs)
        self.doc.refreshProjection()

    def apply_upscale(self):
        path = self.opt['new_img']
        self.save_img(path)
        response = self.upscale(path)
        output = response['output']
        print(f"Getting images: {output}")
        self.insert_img(output)
        self.clear_temp_images(path, [output])
        self.doc.refreshProjection()

    def create_mask_layer_internal(self):
        try:
            if self.selection is not None:
                self.app.action('add_new_transparency_mask').trigger()
                print(f"created mask layer")
                self.doc.setSelection(self.selection)
        finally:
            self.working = False

    def create_mask_layer_workaround(self):
        if self.cfg('create_mask_layer', bool):
            self.working = True
            QTimer.singleShot(self.cfg('workaround_timeout', int), lambda: self.create_mask_layer_internal())

    def clear_temp_images(self, input_file, output_files):
        if self.cfg('delete_temp_files', bool):
            if input_file is not None:
                os.remove(input_file)
            for file in output_files:
                os.remove(file)

    # Actions
    def action_txt2img(self):
        if self.working:
            pass
        self.update_config()
        self.apply_txt2img()
        self.create_mask_layer_workaround()

    def action_img2img(self):
        if self.working:
            pass
        self.update_config()
        self.apply_img2img()
        self.create_mask_layer_workaround()

    def action_upscale(self):
        if self.working:
            pass
        self.update_config()
        self.apply_upscale()
        self.create_mask_layer_workaround()


script = Script()


# Actions for Hotkeys
class MyExtension(Extension):

    def __init__(self, parent):
        # This is initialising the parent, always important when subclassing.
        super().__init__(parent)

    def setup(self):
        pass

    def createActions(self, window):
        txt2img_action = window.createAction("txt2img", "Apply txt2img transform", "tools/scripts")
        txt2img_action.triggered.connect(
            lambda: script.action_txt2img()
        )
        img2img_action = window.createAction("img2img", "Apply img2img transform", "tools/scripts")
        img2img_action.triggered.connect(
            lambda: script.action_img2img()
        )
        upscale_x_action = window.createAction("upscale_x", "Apply upscale transform", "tools/scripts")
        upscale_x_action.triggered.connect(
            lambda: script.action_upscale()
        )


# Interface

class KritaSDPluginDocker(DockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SD Plugin")
        self.create_interface()

        self.init_txt2img_interface()
        self.init_img2img_interface()
        self.init_upscale_interface()
        self.init_config_interface()

        self.connect_txt2img_interface()
        self.connect_img2img_interface()
        self.connect_upscale_interface()
        self.connect_config_interface()

        self.setWidget(self.widget)

    def create_interface(self):
        self.create_txt2img_interface()
        self.create_img2img_interface()
        self.create_upscale_interface()
        self.create_config_interface()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.txt2img_widget, "Txt2Img")
        self.tabs.addTab(self.img2img_widget, "Img2Img")
        self.tabs.addTab(self.upscale_widget, "Upscale")
        self.tabs.addTab(self.config_widget, "Config")

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tabs)
        self.widget = QWidget(self)
        self.widget.setLayout(self.layout)

    def create_txt2img_interface(self):
        self.txt2img_prompt_label = QLabel("Prompt:")
        self.txt2img_prompt_text = QPlainTextEdit()
        self.txt2img_prompt_text.setPlaceholderText("krita_config.yaml value will be used")
        self.txt2img_prompt_layout = QHBoxLayout()
        self.txt2img_prompt_layout.addWidget(self.txt2img_prompt_label)
        self.txt2img_prompt_layout.addWidget(self.txt2img_prompt_text)

        self.txt2img_sampler_name_label = QLabel("Sampler:")
        self.txt2img_sampler_name = QComboBox()
        self.txt2img_sampler_name.addItems(samplers)
        self.txt2img_sampler_name_layout = QHBoxLayout()
        self.txt2img_sampler_name_layout.addWidget(self.txt2img_sampler_name_label)
        self.txt2img_sampler_name_layout.addWidget(self.txt2img_sampler_name)

        self.txt2img_ddim_steps_label = QLabel("Ddim steps:")
        self.txt2img_ddim_steps = QSpinBox()
        self.txt2img_ddim_steps.setMinimum(1)
        self.txt2img_ddim_steps.setMaximum(250)
        self.txt2img_ddim_steps_layout = QHBoxLayout()
        self.txt2img_ddim_steps_layout.addWidget(self.txt2img_ddim_steps_label)
        self.txt2img_ddim_steps_layout.addWidget(self.txt2img_ddim_steps)

        self.txt2img_cfg_scale_label = QLabel("Cfg scale:")
        self.txt2img_cfg_scale = QDoubleSpinBox()
        self.txt2img_cfg_scale.setMinimum(1.0)
        self.txt2img_cfg_scale.setMaximum(30.0)
        self.txt2img_cfg_scale.setSingleStep(0.5)
        self.txt2img_cfg_scale_layout = QHBoxLayout()
        self.txt2img_cfg_scale_layout.addWidget(self.txt2img_cfg_scale_label)
        self.txt2img_cfg_scale_layout.addWidget(self.txt2img_cfg_scale)

        self.txt2img_batch_count_label = QLabel("Batch count:")
        self.txt2img_batch_count = QSpinBox()
        self.txt2img_batch_count.setMinimum(1)
        self.txt2img_batch_count.setMaximum(250)

        self.txt2img_batch_size_label = QLabel("Batch size:")
        self.txt2img_batch_size = QSpinBox()
        self.txt2img_batch_size.setMinimum(1)
        self.txt2img_batch_size.setMaximum(8)

        self.txt2img_batch_layout = QHBoxLayout()
        self.txt2img_batch_layout.addWidget(self.txt2img_batch_count_label)
        self.txt2img_batch_layout.addWidget(self.txt2img_batch_count)
        self.txt2img_batch_layout.addWidget(self.txt2img_batch_size_label)
        self.txt2img_batch_layout.addWidget(self.txt2img_batch_size)

        self.txt2img_base_size_label = QLabel("Base size:")
        self.txt2img_base_size = QSpinBox()
        self.txt2img_base_size.setMinimum(64)
        self.txt2img_base_size.setMaximum(2048)
        self.txt2img_base_size.setSingleStep(64)

        self.txt2img_max_size_label = QLabel("Max size:")
        self.txt2img_max_size = QSpinBox()
        self.txt2img_max_size.setMinimum(64)
        self.txt2img_max_size.setMaximum(2048)
        self.txt2img_max_size.setSingleStep(64)

        self.txt2img_size_layout = QHBoxLayout()
        self.txt2img_size_layout.addWidget(self.txt2img_base_size_label)
        self.txt2img_size_layout.addWidget(self.txt2img_base_size)
        self.txt2img_size_layout.addWidget(self.txt2img_max_size_label)
        self.txt2img_size_layout.addWidget(self.txt2img_max_size)

        self.txt2img_seed_label = QLabel("Seed:")
        self.txt2img_seed = QLineEdit()
        self.txt2img_seed.setPlaceholderText("Random")
        self.txt2img_seed_layout = QHBoxLayout()
        self.txt2img_seed_layout.addWidget(self.txt2img_seed_label)
        self.txt2img_seed_layout.addWidget(self.txt2img_seed)

        self.txt2img_start_button = QPushButton("Apply txt2img")
        self.txt2img_button_layout = QHBoxLayout()
        self.txt2img_button_layout.addWidget(self.txt2img_start_button)

        self.txt2img_layout = QVBoxLayout()
        self.txt2img_layout.addLayout(self.txt2img_prompt_layout)
        self.txt2img_layout.addLayout(self.txt2img_sampler_name_layout)
        self.txt2img_layout.addLayout(self.txt2img_ddim_steps_layout)
        self.txt2img_layout.addLayout(self.txt2img_cfg_scale_layout)
        self.txt2img_layout.addLayout(self.txt2img_batch_layout)
        self.txt2img_layout.addLayout(self.txt2img_size_layout)
        self.txt2img_layout.addLayout(self.txt2img_seed_layout)
        self.txt2img_layout.addLayout(self.txt2img_button_layout)
        self.txt2img_layout.addStretch()

        self.txt2img_widget = QWidget()
        self.txt2img_widget.setLayout(self.txt2img_layout)

    def init_txt2img_interface(self):
        self.txt2img_prompt_text.setPlainText(script.cfg('txt2img_prompt', str))
        self.txt2img_sampler_name.setCurrentIndex(script.cfg('txt2img_sampler', int))
        self.txt2img_ddim_steps.setValue(script.cfg('txt2img_ddim_steps', int))
        self.txt2img_cfg_scale.setValue(script.cfg('txt2img_cfg_scale', float))
        self.txt2img_batch_count.setValue(script.cfg('txt2img_batch_count', int))
        self.txt2img_batch_size.setValue(script.cfg('txt2img_batch_size', int))
        self.txt2img_base_size.setValue(script.cfg('txt2img_base_size', int))
        self.txt2img_max_size.setValue(script.cfg('txt2img_max_size', int))
        self.txt2img_seed.setText(script.cfg('txt2img_seed', str))

    def connect_txt2img_interface(self):
        self.txt2img_prompt_text.textChanged.connect(
            lambda: script.set_cfg("txt2img_prompt",
                                   re.sub(r'\n', ', ', self.txt2img_prompt_text.toPlainText().strip()))
        )
        self.txt2img_sampler_name.currentIndexChanged.connect(
            partial(script.set_cfg, "txt2img_sampler")
        )
        self.txt2img_ddim_steps.valueChanged.connect(
            partial(script.set_cfg, "txt2img_ddim_steps")
        )
        self.txt2img_cfg_scale.valueChanged.connect(
            partial(script.set_cfg, "txt2img_cfg_scale")
        )
        self.txt2img_batch_count.valueChanged.connect(
            partial(script.set_cfg, "txt2img_batch_count")
        )
        self.txt2img_batch_size.valueChanged.connect(
            partial(script.set_cfg, "txt2img_batch_size")
        )
        self.txt2img_base_size.valueChanged.connect(
            partial(script.set_cfg, "txt2img_base_size")
        )
        self.txt2img_max_size.valueChanged.connect(
            partial(script.set_cfg, "txt2img_max_size")
        )
        self.txt2img_seed.textChanged.connect(
            partial(script.set_cfg, "txt2img_seed")
        )
        self.txt2img_start_button.released.connect(
            lambda: script.action_txt2img()
        )

    def create_img2img_interface(self):
        self.img2img_prompt_label = QLabel("Prompt:")
        self.img2img_prompt_text = QPlainTextEdit()
        self.img2img_prompt_text.setPlaceholderText("krita_config.yaml value will be used")
        self.img2img_prompt_layout = QHBoxLayout()
        self.img2img_prompt_layout.addWidget(self.img2img_prompt_label)
        self.img2img_prompt_layout.addWidget(self.img2img_prompt_text)

        self.img2img_sampler_name_label = QLabel("Sampler:")
        self.img2img_sampler_name = QComboBox()
        self.img2img_sampler_name.addItems(samplers)
        self.img2img_sampler_name_layout = QHBoxLayout()
        self.img2img_sampler_name_layout.addWidget(self.img2img_sampler_name_label)
        self.img2img_sampler_name_layout.addWidget(self.img2img_sampler_name)

        self.img2img_ddim_steps_label = QLabel("Ddim steps:")
        self.img2img_ddim_steps = QSpinBox()
        self.img2img_ddim_steps.setMinimum(1)
        self.img2img_ddim_steps.setMaximum(250)
        self.img2img_ddim_steps_layout = QHBoxLayout()
        self.img2img_ddim_steps_layout.addWidget(self.img2img_ddim_steps_label)
        self.img2img_ddim_steps_layout.addWidget(self.img2img_ddim_steps)

        self.img2img_cfg_scale_label = QLabel("Cfg scale:")
        self.img2img_cfg_scale = QDoubleSpinBox()
        self.img2img_cfg_scale.setMinimum(1.0)
        self.img2img_cfg_scale.setMaximum(30.0)
        self.img2img_cfg_scale.setSingleStep(0.5)
        self.img2img_cfg_scale_layout = QHBoxLayout()
        self.img2img_cfg_scale_layout.addWidget(self.img2img_cfg_scale_label)
        self.img2img_cfg_scale_layout.addWidget(self.img2img_cfg_scale)

        self.img2img_denoising_strength_label = QLabel("Denoising strength:")
        self.img2img_denoising_strength = QDoubleSpinBox()
        self.img2img_denoising_strength.setMinimum(0.0)
        self.img2img_denoising_strength.setMaximum(1.0)
        self.img2img_denoising_strength.setSingleStep(0.01)
        self.img2img_denoising_strength_layout = QHBoxLayout()
        self.img2img_denoising_strength_layout.addWidget(self.img2img_denoising_strength_label)
        self.img2img_denoising_strength_layout.addWidget(self.img2img_denoising_strength)

        self.img2img_batch_count_label = QLabel("Batch count:")
        self.img2img_batch_count = QSpinBox()
        self.img2img_batch_count.setMinimum(1)
        self.img2img_batch_count.setMaximum(250)

        self.img2img_batch_size_label = QLabel("Batch size:")
        self.img2img_batch_size = QSpinBox()
        self.img2img_batch_size.setMinimum(1)
        self.img2img_batch_size.setMaximum(8)

        self.img2img_batch_layout = QHBoxLayout()
        self.img2img_batch_layout.addWidget(self.img2img_batch_count_label)
        self.img2img_batch_layout.addWidget(self.img2img_batch_count)
        self.img2img_batch_layout.addWidget(self.img2img_batch_size_label)
        self.img2img_batch_layout.addWidget(self.img2img_batch_size)

        self.img2img_base_size_label = QLabel("Base size:")
        self.img2img_base_size = QSpinBox()
        self.img2img_base_size.setMinimum(64)
        self.img2img_base_size.setMaximum(2048)
        self.img2img_base_size.setSingleStep(64)

        self.img2img_max_size_label = QLabel("Max size:")
        self.img2img_max_size = QSpinBox()
        self.img2img_max_size.setMinimum(64)
        self.img2img_max_size.setMaximum(2048)
        self.img2img_max_size.setSingleStep(64)

        self.img2img_size_layout = QHBoxLayout()
        self.img2img_size_layout.addWidget(self.img2img_base_size_label)
        self.img2img_size_layout.addWidget(self.img2img_base_size)
        self.img2img_size_layout.addWidget(self.img2img_max_size_label)
        self.img2img_size_layout.addWidget(self.img2img_max_size)

        self.img2img_seed_label = QLabel("Seed:")
        self.img2img_seed = QLineEdit()
        self.img2img_seed.setPlaceholderText("Random")
        self.img2img_seed_layout = QHBoxLayout()
        self.img2img_seed_layout.addWidget(self.img2img_seed_label)
        self.img2img_seed_layout.addWidget(self.img2img_seed)

        self.img2img_start_button = QPushButton("Apply img2img")
        self.img2img_button_layout = QHBoxLayout()
        self.img2img_button_layout.addWidget(self.img2img_start_button)

        self.img2img_layout = QVBoxLayout()
        self.img2img_layout.addLayout(self.img2img_prompt_layout)
        self.img2img_layout.addLayout(self.img2img_sampler_name_layout)
        self.img2img_layout.addLayout(self.img2img_ddim_steps_layout)
        self.img2img_layout.addLayout(self.img2img_cfg_scale_layout)
        self.img2img_layout.addLayout(self.img2img_denoising_strength_layout)
        self.img2img_layout.addLayout(self.img2img_batch_layout)
        self.img2img_layout.addLayout(self.img2img_size_layout)
        self.img2img_layout.addLayout(self.img2img_seed_layout)
        self.img2img_layout.addLayout(self.img2img_button_layout)
        self.img2img_layout.addStretch()

        self.img2img_widget = QWidget()
        self.img2img_widget.setLayout(self.img2img_layout)

    def init_img2img_interface(self):
        self.img2img_prompt_text.setPlainText(script.cfg('img2img_prompt', str))
        self.img2img_sampler_name.setCurrentIndex(script.cfg('img2img_sampler', int))
        self.img2img_ddim_steps.setValue(script.cfg('img2img_ddim_steps', int))
        self.img2img_cfg_scale.setValue(script.cfg('img2img_cfg_scale', float))
        self.img2img_denoising_strength.setValue(script.cfg('img2img_denoising_strength', float))
        self.img2img_batch_count.setValue(script.cfg('img2img_batch_count', int))
        self.img2img_batch_size.setValue(script.cfg('img2img_batch_size', int))
        self.img2img_base_size.setValue(script.cfg('img2img_base_size', int))
        self.img2img_max_size.setValue(script.cfg('img2img_max_size', int))
        self.img2img_seed.setText(script.cfg('img2img_seed', str))

    def connect_img2img_interface(self):
        self.img2img_prompt_text.textChanged.connect(
            lambda: script.set_cfg("img2img_prompt",
                                   re.sub(r'\n', ', ', self.img2img_prompt_text.toPlainText().strip()))
        )
        self.img2img_sampler_name.currentIndexChanged.connect(
            partial(script.set_cfg, "img2img_sampler")
        )
        self.img2img_ddim_steps.valueChanged.connect(
            partial(script.set_cfg, "img2img_ddim_steps")
        )
        self.img2img_cfg_scale.valueChanged.connect(
            partial(script.set_cfg, "img2img_cfg_scale")
        )
        self.img2img_denoising_strength.valueChanged.connect(
            partial(script.set_cfg, "img2img_denoising_strength")
        )
        self.img2img_batch_count.valueChanged.connect(
            partial(script.set_cfg, "img2img_batch_count")
        )
        self.img2img_batch_size.valueChanged.connect(
            partial(script.set_cfg, "img2img_batch_size")
        )
        self.img2img_base_size.valueChanged.connect(
            partial(script.set_cfg, "img2img_base_size")
        )
        self.img2img_max_size.valueChanged.connect(
            partial(script.set_cfg, "img2img_max_size")
        )
        self.img2img_seed.textChanged.connect(
            partial(script.set_cfg, "img2img_seed")
        )
        self.img2img_start_button.released.connect(
            lambda: script.action_img2img()
        )

    def create_upscale_interface(self):
        self.upscale_sampler_name_label = QLabel("RealESRGAN model:")
        self.upscale_sampler_name = QComboBox()
        self.upscale_sampler_name.addItems(realesrgan_models)
        self.upscale_model_name_layout = QHBoxLayout()
        self.upscale_model_name_layout.addWidget(self.upscale_sampler_name_label)
        self.upscale_model_name_layout.addWidget(self.upscale_sampler_name)

        self.upscale_base_size_label = QLabel("Base size:")
        self.upscale_base_size = QSpinBox()
        self.upscale_base_size.setMinimum(64)
        self.upscale_base_size.setMaximum(2048)
        self.upscale_base_size.setSingleStep(64)

        self.upscale_max_size_label = QLabel("Max size:")
        self.upscale_max_size = QSpinBox()
        self.upscale_max_size.setMinimum(64)
        self.upscale_max_size.setMaximum(2048)
        self.upscale_max_size.setSingleStep(64)

        self.upscale_size_layout = QHBoxLayout()
        self.upscale_size_layout.addWidget(self.upscale_base_size_label)
        self.upscale_size_layout.addWidget(self.upscale_base_size)
        self.upscale_size_layout.addWidget(self.upscale_max_size_label)
        self.upscale_size_layout.addWidget(self.upscale_max_size)

        self.upscale_start_button = QPushButton("Apply upscale with RealESRGAN")
        self.upscale_button_layout = QHBoxLayout()
        self.upscale_button_layout.addWidget(self.upscale_start_button)

        self.upscale_layout = QVBoxLayout()
        self.upscale_layout.addLayout(self.upscale_model_name_layout)
        self.upscale_layout.addLayout(self.upscale_size_layout)
        self.upscale_layout.addLayout(self.upscale_button_layout)
        self.upscale_layout.addStretch()

        self.upscale_widget = QWidget()
        self.upscale_widget.setLayout(self.upscale_layout)

    def init_upscale_interface(self):
        self.upscale_sampler_name.setCurrentIndex(script.cfg('upscale_realesrgan_model', int))
        self.upscale_base_size.setValue(script.cfg('upscale_base_size', int))
        self.upscale_max_size.setValue(script.cfg('upscale_max_size', int))

    def connect_upscale_interface(self):
        self.upscale_sampler_name.currentIndexChanged.connect(
            partial(script.set_cfg, "upscale_realesrgan_model")
        )
        self.upscale_base_size.valueChanged.connect(
            partial(script.set_cfg, "upscale_base_size")
        )
        self.upscale_max_size.valueChanged.connect(
            partial(script.set_cfg, "upscale_max_size")
        )
        self.upscale_start_button.released.connect(
            lambda: script.action_img2img()
        )

    def create_config_interface(self) -> QWidget:
        self.config_base_url_label = QLabel("Backend url (only local now):")
        self.config_base_url = QLineEdit()
        self.config_base_url_reset = QPushButton("Default")
        self.config_base_url_layout = QHBoxLayout()
        self.config_base_url_layout.addWidget(self.config_base_url)
        self.config_base_url_layout.addWidget(self.config_base_url_reset)

        self.config_just_use_yaml = QCheckBox("Use only YAML config, ignore these properties")
        self.config_just_use_yaml.setTristate(False)
        self.config_create_mask_layer = QCheckBox("Create transparency mask layer from selection")
        self.config_create_mask_layer.setTristate(False)
        self.config_delete_temp_files = QCheckBox("Automatically delete temporary image files")
        self.config_delete_temp_files.setTristate(False)

        self.config_restore_defaults = QPushButton("Restore Defaults")

        self.config_layout = QVBoxLayout()
        self.config_layout.addWidget(self.config_base_url_label)
        self.config_layout.addLayout(self.config_base_url_layout)
        self.config_layout.addWidget(self.config_just_use_yaml)
        self.config_layout.addWidget(self.config_create_mask_layer)
        self.config_layout.addWidget(self.config_delete_temp_files)
        self.config_layout.addWidget(self.config_restore_defaults)
        self.config_layout.addStretch()

        self.config_widget = QWidget()
        self.config_widget.setLayout(self.config_layout)
        return self.config_widget

    def init_config_interface(self):
        self.config_base_url.setText(script.cfg('base_url', str))
        self.config_just_use_yaml.setCheckState(
            Qt.CheckState.Checked if script.cfg('just_use_yaml', bool) else Qt.CheckState.Unchecked)
        self.config_create_mask_layer.setCheckState(
            Qt.CheckState.Checked if script.cfg('create_mask_layer', bool) else Qt.CheckState.Unchecked)
        self.config_delete_temp_files.setCheckState(
            Qt.CheckState.Checked if script.cfg('delete_temp_files', bool) else Qt.CheckState.Unchecked)

    def connect_config_interface(self):
        self.config_base_url.textChanged.connect(
            partial(script.set_cfg, "base_url")
        )
        self.config_base_url_reset.released.connect(
            lambda: self.config_base_url.setText(default_url)
        )
        self.config_just_use_yaml.toggled.connect(
            partial(script.set_cfg, "just_use_yaml")
        )
        self.config_create_mask_layer.toggled.connect(
            partial(script.set_cfg, "create_mask_layer")
        )
        self.config_delete_temp_files.toggled.connect(
            partial(script.set_cfg, "delete_temp_files")
        )
        self.config_restore_defaults.released.connect(
            lambda: self.restore_defaults()
        )

    def restore_defaults(self):
        script.restore_defaults()
        self.init_txt2img_interface()
        self.init_img2img_interface()
        self.init_upscale_interface()
        self.init_upscale_interface()

    def canvasChanged(self, canvas):
        pass


# And add the extension to Krita's list of extensions:
Krita.instance().addExtension(MyExtension(Krita.instance()))
Krita.instance().addDockWidgetFactory(
    DockWidgetFactory("krita_diff", DockWidgetFactoryBase.DockRight, KritaSDPluginDocker))
