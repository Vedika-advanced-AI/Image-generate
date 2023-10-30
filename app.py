from app_settings import AppSettings
from utils import show_system_info
import constants
from argparse import ArgumentParser
from context import Context
from constants import APP_VERSION, LCM_DEFAULT_MODEL_OPENVINO
from models.interface_types import InterfaceType
from constants import DEVICE

parser = ArgumentParser(description=f"FAST SD CPU {constants.APP_VERSION}")
parser.add_argument(
    "-s",
    "--share",
    action="store_true",
    help="Create sharable link(Web UI)",
    required=False,
)
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument(
    "-g",
    "--gui",
    action="store_true",
    help="Start desktop GUI",
)
group.add_argument(
    "-w",
    "--webui",
    action="store_true",
    help="Start Web UI",
)
group.add_argument(
    "-v",
    "--version",
    action="store_true",
    help="Version",
)
parser.add_argument(
    "--lcm_model_id",
    type=str,
    help="Model ID or path,Default SimianLuo/LCM_Dreamshaper_v7",
    default="SimianLuo/LCM_Dreamshaper_v7",
)
parser.add_argument(
    "--prompt",
    type=str,
    help="Describe the image you want to generate",
)
parser.add_argument(
    "--image_height",
    type=int,
    help="Height of the image",
    default=512,
)
parser.add_argument(
    "--image_width",
    type=int,
    help="Width of the image",
    default=512,
)
parser.add_argument(
    "--inference_steps",
    type=int,
    help="Number of steps,default : 4",
    default=4,
)
parser.add_argument(
    "--guidance_scale",
    type=int,
    help="Guidance scale,default : 8.0",
    default=8.0,
)

parser.add_argument(
    "--number_of_images",
    type=int,
    help="Number of images to generate ,default : 1",
    default=1,
)
parser.add_argument(
    "--seed",
    type=int,
    help="Seed,default : -1 (disabled) ",
    default=-1,
)
parser.add_argument(
    "--use_openvino",
    action="store_true",
    help="Use OpenVINO model",
)

parser.add_argument(
    "--use_offline_model",
    action="store_true",
    help="Use offline model",
)
parser.add_argument(
    "--use_safety_checker",
    action="store_false",
    help="Use safety checker",
)

parser.add_argument(
    "-i",
    "--interactive",
    action="store_true",
    help="Interactive CLI mode",
)

args = parser.parse_args()

if args.version:
    print(APP_VERSION)
    exit()

parser.print_help()
show_system_info()
print(f"Using device : {constants.DEVICE}")

app_settings = AppSettings()
app_settings.load()


start_webui(
        app_settings,
        args.share,)