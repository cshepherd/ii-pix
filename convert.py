"""Image converter to Apple II Double Hi-Res format."""

import argparse
import numpy as np

import convert_hgr as convert_hgr_py
import convert_dhr as convert_dhr_py
import convert_shr as convert_shr_py
import dither_pattern
import image as image_py
import palette as palette_py
import screen as screen_py


# TODO:
#  - support additional graphics modes (easiest --> hardest):
#    - LR/DLR
#    - SHR 3200
#    - SHR 640
#    - HGR


def add_common_args(parser):
    parser.add_argument("input", type=str, help="Input image file to process.")
    parser.add_argument("output", type=str, help="Output file for converted "
                                                 "Apple II image.")
    parser.add_argument(
        '--show-input', action=argparse.BooleanOptionalAction, default=False,
        help="Whether to show the input image before conversion.")
    parser.add_argument(
        '--show-output', action=argparse.BooleanOptionalAction, default=True,
        help="Whether to show the output image after conversion.")
    parser.add_argument(
        '--save-preview', action=argparse.BooleanOptionalAction, default=True,
        help='Whether to save a .PNG rendering of the output image (default: '
             'True)'
    )
    parser.add_argument(
        '--verbose', action=argparse.BooleanOptionalAction,
        default=False, help="Show progress during conversion")
    parser.add_argument(
        '--gamma-correct', type=float, default=2.4,
        help='Gamma-correct image by this value (default: 2.4)'
    )


def add_dhr_hgr_args(parser):
    parser.add_argument(
        '--dither', type=str, choices=list(dither_pattern.PATTERNS.keys()),
        default=dither_pattern.DEFAULT_PATTERN,
        help="Error distribution pattern to apply when dithering (default: "
             + dither_pattern.DEFAULT_PATTERN + ")")
    parser.add_argument(
        '--palette', type=str, choices=list(set(palette_py.PALETTES.keys())),
        default=palette_py.DEFAULT_PALETTE,
        help='RGB colour palette to dither to.  "ntsc" blends colours over 8 '
             'pixels and gives better image quality on targets that '
             'use/emulate NTSC, but can be substantially slower.  Other '
             'palettes determine colours based on 4 pixel sequences '
             '(default: ' + palette_py.DEFAULT_PALETTE + ")")
    parser.add_argument(
        '--show-palette', type=str, choices=list(palette_py.PALETTES.keys()),
        help="RGB colour palette to use when --show_output (default: "
             "value of --palette)")


def validate_lookahead(arg: int) -> int:
    try:
        int_arg = int(arg)
    except Exception:
        raise argparse.ArgumentTypeError("--lookahead must be an integer")
    if int_arg < 1:
        raise argparse.ArgumentTypeError("--lookahead must be at least 1")
    return int_arg


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    # Hi-res
    hgr_parser = subparsers.add_parser("hgr")
    add_common_args(hgr_parser)
    add_dhr_hgr_args(hgr_parser)
    hgr_parser.add_argument(
        '--error_fraction', type=float, default = 0.7,
        help="Fraction of quantization error to distribute to neighbouring "
             "pixels according to dither pattern"
    )
    hgr_parser.set_defaults(func=convert_hgr)

    # Double Hi-res
    dhr_parser = subparsers.add_parser("dhr")
    add_common_args(dhr_parser)
    add_dhr_hgr_args(dhr_parser)
    dhr_parser.add_argument(
        "--lookahead", type=validate_lookahead, default=8,
        help=("How many pixels to look ahead to compensate for NTSC colour "
              "artifacts (default: 8)"))
    dhr_parser.set_defaults(func=convert_dhr)

    # Double Hi-Res mono
    dhr_mono_parser = subparsers.add_parser("dhr_mono")
    add_common_args(dhr_mono_parser)
    dhr_mono_parser.set_defaults(func=convert_dhr_mono)

    # Super Hi-Res 320x200
    shr_parser = subparsers.add_parser("shr")
    add_common_args(shr_parser)
    shr_parser.add_argument(
        '--dither', type=str, choices=['floyd-steinberg', 'jarvis', 'none'],
        default='floyd-steinberg',
        help='Dithering algorithm to use, or "none" for no dithering '
             '(default: floyd-steinberg)')
    shr_parser.add_argument(
        '--fixed-colours', type=int, default=0,
        help='How many colours to fix as identical across all 16 SHR palettes '
             '(default: 0)'
    )
    shr_parser.add_argument(
        '--reserve-colours', type=int, default=0,
        help='How many palette entries per scanline to reserve for sprites. '
             'The background image will use 16 - N colours per palette, '
             'leaving entries N..15 free. (default: 0)'
    )
    shr_parser.add_argument(
        '--show-final-score', action=argparse.BooleanOptionalAction,
        default=False, help='Whether to output the final image quality score '
                            '(default: False)'
    )
    shr_parser.add_argument(
        '--save-intermediate', action=argparse.BooleanOptionalAction,
        default=False, help='Whether to save each intermediate iteration, '
                            'or just the final image (default: False)'
    )
    shr_parser.add_argument(
        '--palette-tolerance', type=float, default=0,
        help='Merge palette colours across all palettes whose Euclidean '
             'distance in 4-bit RGB space is within this threshold, snapping '
             'them to a single representative value.  Useful for sprites '
             'that need identical colour values across palettes.  '
             'A value of 1 merges colours that differ by at most 1 step in '
             'a single channel (default: 0, disabled)')
    shr_parser.add_argument(
        '--palette-order', type=str, choices=['none', 'hue'],
        default='none',
        help='Reorder palette entries before output.  "hue" sorts by hue '
             'so the same colour tends to land at the same index across '
             'palettes (default: none)')
    shr_parser.add_argument(
        '--palette-file', type=str, default=None,
        help='Path to an existing .SHR file whose palettes will be reused '
             'exactly.  ii-pix will only choose which palette and colours '
             'to use per scanline/pixel. (default: None)'
    )
    shr_parser.add_argument(
        '--palette-and-scb-file', type=str, default=None,
        help='Path to an existing .SHR file whose palettes and scanline '
             'control bytes (palette-per-line assignments) will be reused '
             'exactly.  ii-pix will only choose which colour to use per '
             'pixel. (default: None)'
    )
    shr_parser.add_argument(
        '--one-palette', action='store_true', default=False,
        help='Use a single 16-colour palette for the entire image, with '
             'the same palette assigned to every scanline. (default: False)'
    )
    shr_parser.add_argument(
        '--no-upscale', action='store_true', default=False,
        help='If the input image is smaller than 320x200, do not scale it up. '
             'Instead, place it at the top-left and fill the remaining area '
             'with black. (default: False)'
    )
    shr_parser.add_argument(
        '--3200', dest='shr_3200', action='store_true', default=False,
        help='Convert to a 3200-colour image, i.e. an independent 16-colour '
             'palette for each of the 200 scanlines.  Scanline control bytes '
             'number the palettes from 15 down to 0 repeatedly (the final 8 '
             'scanlines continue the descending order), for display code '
             'that rewrites the 16 hardware palettes on the fly from SCB '
             'interrupts.  The output .SHR file contains the palettes for '
             'the first 16 scanlines; all 200 palettes are also written to '
             '<output>.palettes. (default: False)'
    )
    shr_parser.set_defaults(func=convert_shr)
    args = parser.parse_args()
    args.func(args)


def prepare_image(image_filename: str, show_input: bool, screen,
                  gamma_correct: float,
                  no_upscale: bool = False) -> np.ndarray:
    # Open and resize source image
    image = image_py.open(image_filename)
    if show_input:
        image_py.resize(image, screen.X_RES, screen.Y_RES * 2,
                        srgb_output=True).show()

    if no_upscale and image.width <= screen.X_RES and \
            image.height <= screen.Y_RES:
        # Convert to linear RGB at original size, then pad to screen resolution
        from PIL import Image
        linear = image_py.resize(image, image.width, image.height,
                                 gamma=gamma_correct)
        padded = Image.new(linear.mode, (screen.X_RES, screen.Y_RES), (0, 0, 0))
        padded.paste(linear, (0, 0))
        return padded

    return image_py.resize(image, screen.X_RES, screen.Y_RES,
                           gamma=gamma_correct)


def convert_hgr(args):
    palette = palette_py.PALETTES[args.palette]()
    screen = screen_py.HGRNTSCScreen(palette)
    image = prepare_image(args.input, args.show_input, screen,
                          args.gamma_correct)
    convert_hgr_py.convert(screen, image, args)


def convert_dhr(args):
    palette = palette_py.PALETTES[args.palette]()
    screen = screen_py.DHGRNTSCScreen(palette)
    image = prepare_image(args.input, args.show_input, screen,
                          args.gamma_correct)
    convert_dhr_py.convert(screen, image, args)


def convert_dhr_mono(args):
    screen = screen_py.DHGRScreen()
    image = prepare_image(args.input, args.show_input, screen,
                          args.gamma_correct)
    convert_dhr_py.convert_mono(screen, image, args)


def convert_shr(args):
    screen = screen_py.SHR320Screen()
    image = prepare_image(args.input, args.show_input, screen,
                          args.gamma_correct,
                          no_upscale=args.no_upscale)
    convert_shr_py.convert(screen, image, args)


if __name__ == "__main__":
    main()
