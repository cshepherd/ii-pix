import colorsys
from collections import defaultdict
import os.path
import random
from typing import Tuple

from PIL import Image
import colour
import numpy as np
from sklearn import cluster

from os import environ

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

import dither_shr as dither_shr_pyx
import image as image_py


class ClusterPalette:
    def __init__(
            self, image: np.ndarray, rgb12_iigs_to_cam16ucs, rgb24_to_cam16ucs,
            fixed_colours=0, reserve_colours=0, dither='floyd-steinberg'):

        # Conversion matrix from 12-bit //gs RGB colour space to CAM16UCS
        # colour space
        self._rgb12_iigs_to_cam16ucs = rgb12_iigs_to_cam16ucs

        # Conversion matrix from 24-bit linear RGB colour space to CAM16UCS
        # colour space
        self._rgb24_to_cam16ucs = rgb24_to_cam16ucs

        # How many palette entries to reserve (for e.g. sprites).  Background
        # image uses palette entries 0 .. colours_per_palette-1; entries
        # colours_per_palette .. 15 are left zeroed for the caller to fill.
        self._reserve_colours = reserve_colours
        self._colours_per_palette = 16 - reserve_colours

        # Dithering algorithm: 'floyd-steinberg', 'jarvis', or 'none'
        self._dither = dither

        # Preprocessed source image in 24-bit linear RGB colour space.  We
        # first dither the source image using the full 12-bit //gs RGB colour
        # palette, ignoring SHR palette limitations (i.e. 4096 independent
        # colours for each pixel).  This gives much better results for e.g.
        # solid blocks of colour, which would be dithered inconsistently if
        # targeting the source image directly.
        self._image_rgb = self._perfect_dither(image)

        # Preprocessed source image in CAM16UCS colour space
        self._colours_cam = self._image_colours_cam(self._image_rgb)

        # We fit a palette against the entire image which is used as starting
        # values for fitting the reserved colours in the 16 SHR palettes.
        self._global_palette = np.empty(
            (self._colours_per_palette, 3), dtype=np.uint8)

        # How many image colours to fix identically across all 16 SHR
        # palettes.  These are taken to be the most prevalent colours from
        # _global_palette.
        self._fixed_colours = fixed_colours

        # 16 SHR palettes each of 16 colours, in CAM16UCS colour space.
        # Only the first colours_per_palette entries per palette are used;
        # the rest are zeroed (reserved for sprites).
        self._palettes_cam = np.zeros((16, 16, 3), dtype=np.float32)

        # 16 SHR palettes each of 16 colours, in //gs 4-bit RGB colour space
        self._palettes_rgb = np.zeros((16, 16, 3), dtype=np.uint8)

        # defaultdict(list) mapping palette index to the lines that use this
        # palette
        self._palette_lines = self._init_palette_lines()

    @staticmethod
    def _image_colours_cam(image: Image):
        colours_rgb = np.asarray(image)  # .reshape((-1, 3))
        with colour.utilities.suppress_warnings(colour_usage_warnings=True):
            colours_cam = colour.convert(colours_rgb, "RGB",
                                         "CAM16UCS").astype(np.float32)
        return colours_cam

    def _init_palette_lines(self, init_random=False):
        palette_lines = defaultdict(list)

        if init_random:
            lines = list(range(200))
            random.shuffle(lines)
            idx = 0
            while lines:
                palette_lines[idx].append(lines.pop())
                idx += 1
        else:
            palette_splits = self._equal_palette_splits()
            for i, lh in enumerate(palette_splits):
                l, h = lh
                palette_lines[i].extend(list(range(l, h)))
        return palette_lines

    @staticmethod
    def _equal_palette_splits(palette_height=35):
        # The 16 palettes are striped across consecutive (overlapping) line
        # ranges.  Since nearby lines tend to have similar colours, this has
        # the effect of smoothing out the colour transitions across palettes.

        # If we want to overlap 16 palettes in 200 lines, where each palette
        # has height H and overlaps the previous one by L lines, then the
        # boundaries are at lines:
        #   (0, H), (H-L, 2H-L), (2H-2L, 3H-2L), ..., (15H-15L, 16H - 15L)
        # i.e. 16H - 15L = 200, so for a given palette height H we need to
        # overlap by:
        #   L = (16H - 200)/15

        palette_overlap = (16 * palette_height - 200) / 15

        palette_ranges = []
        for palette_idx in range(16):
            palette_lower = palette_idx * (palette_height - palette_overlap)
            palette_upper = palette_lower + palette_height
            palette_ranges.append((int(np.round(palette_lower)),
                                   int(np.round(palette_upper))))
        return palette_ranges

    def _perfect_dither(self, source_image: np.ndarray):
        """Dither a "perfect" image using the full 12-bit //gs RGB colour
        palette, ignoring restrictions."""

        # Suppress divide by zero warning,
        # https://github.com/colour-science/colour/issues/900
        with colour.utilities.suppress_warnings(python_warnings=True):
            full_palette_linear_rgb = colour.convert(
                self._rgb12_iigs_to_cam16ucs, "CAM16UCS", "RGB").astype(
                np.float32)

        total_image_error, image_rgb = dither_shr_pyx.dither_shr_perfect(
            source_image, self._rgb12_iigs_to_cam16ucs, full_palette_linear_rgb,
            self._rgb24_to_cam16ucs, self._dither)
        # print("Perfect image error:", total_image_error)
        return image_rgb

    def _dither_image(self, palettes_cam):
        # Suppress divide by zero warning,
        # https://github.com/colour-science/colour/issues/900
        with colour.utilities.suppress_warnings(python_warnings=True):
            palettes_linear_rgb = colour.convert(
                palettes_cam, "CAM16UCS", "RGB").astype(np.float32)

        output_4bit, line_to_palette, total_image_error, palette_line_errors = \
            dither_shr_pyx.dither_shr(
                self._image_rgb, palettes_cam, palettes_linear_rgb,
                self._rgb24_to_cam16ucs, self._colours_per_palette,
                self._dither)

        # Update map of palettes to image lines for which the palette was the
        # best match
        palette_lines = defaultdict(list)
        for line, palette in enumerate(line_to_palette):
            palette_lines[palette].append(line)
        self._palette_lines = palette_lines

        self._palette_line_errors = palette_line_errors

        return (output_4bit, line_to_palette, palettes_linear_rgb,
                total_image_error)

    def iterate(self, max_inner_iterations: int,
                max_outer_iterations: int):
        total_image_error = 1e9

        outer_iterations_since_improvement = 0
        while outer_iterations_since_improvement < max_outer_iterations:
            inner_iterations_since_improvement = 0
            self._palette_lines = self._init_palette_lines()

            while inner_iterations_since_improvement < max_inner_iterations:
                # print("Iterations %d" % inner_iterations_since_improvement)
                new_palettes_cam, new_palettes_rgb12_iigs = (
                    self._fit_shr_palettes())

                # Recompute image with proposed palettes and check whether it
                # has lower total image error than our previous best.
                (output_4bit, line_to_palette, palettes_linear_rgb,
                 new_total_image_error) = self._dither_image(new_palettes_cam)

                self._reassign_unused_palettes(
                    line_to_palette, new_palettes_rgb12_iigs)

                if new_total_image_error >= total_image_error:
                    inner_iterations_since_improvement += 1
                    continue

                # We found a globally better set of palettes, so restart the
                # clocks
                inner_iterations_since_improvement = 0
                outer_iterations_since_improvement = -1
                total_image_error = new_total_image_error

                self._palettes_cam = new_palettes_cam
                self._palettes_rgb = new_palettes_rgb12_iigs

                yield (new_total_image_error, output_4bit, line_to_palette,
                       new_palettes_rgb12_iigs, palettes_linear_rgb)
            outer_iterations_since_improvement += 1

    def _fit_shr_palettes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Attempt to find new palettes that locally improve image quality.

        Re-fit a set of 16 palettes from (overlapping) line ranges of the
        source image, using k-means clustering in CAM16-UCS colour space.

        We maintain the total image error for the pixels on which the 16
        palettes are clustered.  A new palette that increases this local
        image error is rejected.

        New palettes that reduce local error cannot be applied immediately
        though, because they may cause an increase in *global* image error
        when dithering.  i.e. they would reduce the overall image quality.

        The current (locally) best palettes are returned and can be applied
        using accept_palettes()

        XXX update
        """
        new_palettes_cam = np.zeros_like(self._palettes_cam)
        new_palettes_rgb12_iigs = np.zeros_like(self._palettes_rgb)

        # Compute a new 16-colour global palette for the entire image,
        # used as the starting center positions for k-means clustering of the
        # individual palettes
        self._fit_global_palette()

        for palette_idx in range(16):
            palette_pixels = (
                self._colours_cam[self._palette_lines[
                                      palette_idx], :, :].reshape(-1, 3))

            n = self._colours_per_palette

            # Fix reserved colours from the global palette.
            initial_centroids = np.copy(self._global_palette)
            pixels_rgb_iigs = dither_shr_pyx.convert_cam16ucs_to_rgb12_iigs(
                palette_pixels)
            seen_colours = set()
            for i in range(self._fixed_colours):
                seen_colours.add(tuple(initial_centroids[i, :]))

            # Pick unique random colours from the sample points for the
            # remaining initial centroids.
            for i in range(self._fixed_colours, n):
                choice = np.random.randint(0, pixels_rgb_iigs.shape[0])
                new_colour = pixels_rgb_iigs[choice, :]
                if tuple(new_colour) in seen_colours:
                    continue
                seen_colours.add(tuple(new_colour))
                initial_centroids[i, :] = new_colour

            # If there are any single colours in our source //gs RGB pixels that
            # represent more than fixed_colour_fraction_threshold of the total,
            # then fix these colours for the palette instead of clustering
            # them.  This reduces artifacting on blocks of colour.
            fixed_colour_fraction_threshold = 0.1
            most_frequent_colours = sorted(list(zip(
                *np.unique(pixels_rgb_iigs, return_counts=True, axis=0))),
                key=lambda kv: kv[1], reverse=True)
            fixed_colours = self._fixed_colours
            for palette_colour, freq in most_frequent_colours:
                if (freq < (palette_pixels.shape[0] *
                            fixed_colour_fraction_threshold)) or (
                        fixed_colours == n):
                    break
                if tuple(palette_colour) not in seen_colours:
                    seen_colours.add(tuple(palette_colour))
                    initial_centroids[fixed_colours, :] = palette_colour
                    fixed_colours += 1

            palette_rgb12_iigs = dither_shr_pyx.k_means_with_fixed_centroids(
                n_clusters=n, n_fixed=fixed_colours,
                samples=palette_pixels,
                initial_centroids=initial_centroids,
                max_iterations=1000,
                rgb12_iigs_to_cam16ucs=self._rgb12_iigs_to_cam16ucs)
            # If the k-means clustering returned fewer than the target number
            # of unique colours, fill out the remainder with the most common
            # pixel colours that have not yet been used.
            #
            # TODO: this seems like an opportunity to do something better -
            #   e.g. forcibly split clusters and iterate the clustering
            palette_rgb12_iigs = self._fill_short_palette(
                palette_rgb12_iigs, most_frequent_colours)

            for i in range(n):
                new_palettes_cam[palette_idx, i, :] = (
                    np.array(dither_shr_pyx.convert_rgb12_iigs_to_cam(
                        self._rgb12_iigs_to_cam16ucs, palette_rgb12_iigs[
                            i]), dtype=np.float32))

            new_palettes_rgb12_iigs[palette_idx, :n, :] = palette_rgb12_iigs

        self._palettes_accepted = False
        return new_palettes_cam, new_palettes_rgb12_iigs

    def _fit_global_palette(self):
        """Compute a palette for the entire image to use as starting point
        for the sub-palettes.  This should help when the image has large
        blocks of colour since the sub-palettes will tend to pick the same
        colours."""

        n = self._colours_per_palette
        clusters = cluster.MiniBatchKMeans(n_clusters=n, max_iter=10000)
        clusters.fit_predict(self._colours_cam.reshape(-1, 3))

        # Dict of {palette idx : frequency count}
        palette_freq = {idx: 0 for idx in range(n)}
        for idx, freq in zip(*np.unique(clusters.labels_, return_counts=True)):
            palette_freq[idx] = freq

        frequency_order = [
            k for k, v in sorted(
                list(palette_freq.items()), key=lambda kv: kv[1], reverse=True)]

        self._global_palette = (
            dither_shr_pyx.convert_cam16ucs_to_rgb12_iigs(
                clusters.cluster_centers_[frequency_order].astype(
                    np.float32)))

    def _fill_short_palette(self, palette_iigs_rgb, most_frequent_colours):
        """Fill out the palette to colours_per_palette unique entries."""

        target = self._colours_per_palette

        # We want to maintain order of insertion so that we respect the
        # ordering of fixed colours in the palette.  Python doesn't have an
        # orderedset but dicts preserve insertion order.
        palette_set = {}
        for palette_entry in palette_iigs_rgb:
            palette_set[tuple(palette_entry)] = True
        if len(palette_set) == target:
            return palette_iigs_rgb

        # Add most frequent image colours that are not yet in the palette
        for palette_colour, freq in most_frequent_colours:
            if tuple(palette_colour) in palette_set:
                continue
            palette_set[tuple(palette_colour)] = True
            if len(palette_set) == target:
                break

        # We couldn't find any more unique colours, fill out with random ones.
        while len(palette_set) < target:
            palette_set[
                tuple(np.random.randint(0, 16, size=3, dtype=np.uint8))] = True

        return np.array(tuple(palette_set.keys()), dtype=np.uint8)

    def _reassign_unused_palettes(self, line_to_palette, palettes_iigs_rgb):
        palettes_used = [False] * 16
        for palette in line_to_palette:
            palettes_used[palette] = True
        best_palette_lines = [v for k, v in sorted(list(zip(
            self._palette_line_errors, range(200))))]

        all_palettes = set()
        for palette_idx, palette_iigs_rgb in enumerate(palettes_iigs_rgb):
            palette_set = set()
            for palette_entry in palette_iigs_rgb:
                palette_set.add(tuple(palette_entry))
            palette_set = frozenset(palette_set)
            if palette_set in all_palettes:
                print("Duplicate palette", palette_idx, palette_set)
                palettes_used[palette_idx] = False

        for palette_idx, palette_used in enumerate(palettes_used):
            if palette_used:
                continue

            # TODO: also remove from old entry
            worst_line = best_palette_lines.pop()
            self._palette_lines[palette_idx] = [worst_line]


def _output_image(screen, output_4bit, line_to_palette, palettes_rgb12_iigs,
                  palettes_linear_rgb, args, output_base, output_ext, seq,
                  canvas=None, total_image_error=None,
                  new_total_image_error=None):
    """Common output logic shared by both conversion modes."""
    if args.verbose and total_image_error is not None:
        print("Improved quality +%f%% (%f)" % (
            (1 - new_total_image_error / total_image_error) * 100,
            new_total_image_error))

    palette_tolerance = getattr(args, 'palette_tolerance', 0)
    if palette_tolerance > 0:
        reserve = getattr(args, 'reserve_colours', 0)
        n_active = 16 - reserve
        tol_sq = palette_tolerance * palette_tolerance

        # Collect every colour and count how often it appears across all
        # palette slots so we can pick the most common as representative.
        colour_freq = {}
        for pal_idx in range(16):
            for c in range(n_active):
                rgb = tuple(int(v) for v in palettes_rgb12_iigs[pal_idx, c, :])
                colour_freq[rgb] = colour_freq.get(rgb, 0) + 1

        # Greedy clustering: iterate colours from most to least frequent.
        # Each colour either joins an existing cluster (if within tolerance
        # of its representative) or starts a new one.
        sorted_colours = sorted(colour_freq, key=colour_freq.get, reverse=True)
        clusters = []  # list of (representative_rgb, set of member rgbs)
        colour_to_rep = {}

        for rgb in sorted_colours:
            best_dist = tol_sq + 1
            best_rep = None
            for rep, _ in clusters:
                d = sum((a - b) ** 2 for a, b in zip(rgb, rep))
                if d < best_dist:
                    best_dist = d
                    best_rep = rep
            if best_dist <= tol_sq:
                colour_to_rep[rgb] = best_rep
                # Add to existing cluster
                for rep, members in clusters:
                    if rep == best_rep:
                        members.add(rgb)
                        break
            else:
                # New cluster — this colour is its own representative
                colour_to_rep[rgb] = rgb
                clusters.append((rgb, {rgb}))

        # Precompute linear RGB for each representative colour.
        rgb12_iigs_to_cam16ucs = np.load(
            os.path.join(os.path.dirname(__file__),
                         "data/rgb12_iigs_to_cam16ucs.npy"))
        rep_linear_rgb = {}
        for rep, _ in clusters:
            cam = np.array(dither_shr_pyx.convert_rgb12_iigs_to_cam(
                rgb12_iigs_to_cam16ucs, np.array(rep, dtype=np.uint8)),
                dtype=np.float32).reshape(1, 3)
            with colour.utilities.suppress_warnings(python_warnings=True):
                rep_linear_rgb[rep] = colour.convert(
                    cam, "CAM16UCS", "RGB").astype(np.float32)[0]

        # Apply: replace every palette entry with its representative.
        for pal_idx in range(16):
            for c in range(n_active):
                rgb = tuple(int(v) for v in palettes_rgb12_iigs[pal_idx, c, :])
                rep = colour_to_rep[rgb]
                if rep != rgb:
                    for i in range(3):
                        palettes_rgb12_iigs[pal_idx, c, i] = rep[i]
                    palettes_linear_rgb[pal_idx, c, :] = rep_linear_rgb[rep]

        if args.verbose:
            merged = sum(1 for rgb, rep in colour_to_rep.items() if rgb != rep)
            print("Palette tolerance: merged %d colours into %d clusters" % (
                merged, len(clusters)))

    palette_order = getattr(args, 'palette_order', 'none')
    if palette_order == 'hue':
        reserve = getattr(args, 'reserve_colours', 0)
        n_active = 16 - reserve

        # Build a canonical ordering from all unique colours across every
        # palette, sorted by (hue, saturation, value).  Each palette's
        # entries are then sorted by their position in this master list,
        # so a colour that appears in multiple palettes lands at the same
        # (or very close) index in each.
        canonical_key = {}  # (r,g,b) -> (h, s, v)
        for pal_idx in range(16):
            for c in range(n_active):
                rgb = tuple(int(v) for v in palettes_rgb12_iigs[pal_idx, c, :])
                if rgb not in canonical_key:
                    h, s, v = colorsys.rgb_to_hsv(
                        rgb[0] / 15, rgb[1] / 15, rgb[2] / 15)
                    canonical_key[rgb] = (h, s, v)

        for pal_idx in range(16):
            # Sort this palette's active entries by canonical HSV key
            keys = []
            for c in range(n_active):
                rgb = tuple(int(v) for v in palettes_rgb12_iigs[pal_idx, c, :])
                keys.append(canonical_key[rgb])
            order = sorted(range(n_active), key=lambda i: keys[i])

            # Build inverse map: order[new] = old, so inv[old] = new
            inv = [0] * n_active
            for new_idx, old_idx in enumerate(order):
                inv[old_idx] = new_idx

            # Reorder palette arrays
            palettes_rgb12_iigs[pal_idx, :n_active, :] = (
                palettes_rgb12_iigs[pal_idx, order, :])
            palettes_linear_rgb[pal_idx, :n_active, :] = (
                palettes_linear_rgb[pal_idx, order, :])

            # Remap pixel indices on lines that use this palette
            for y in range(200):
                if line_to_palette[y] == pal_idx:
                    for x in range(320):
                        old = output_4bit[y, x]
                        if old < n_active:
                            output_4bit[y, x] = inv[old]

    for i in range(16):
        screen.set_palette(i, palettes_rgb12_iigs[i, :, :])

    screen.set_pixels(output_4bit)
    output_rgb = np.empty((200, 320, 3), dtype=np.uint8)
    for i in range(200):
        screen.line_palette[i] = line_to_palette[i]
        output_rgb[i, :, :] = (
                palettes_linear_rgb[line_to_palette[i]][
                    output_4bit[i, :]] * 255
        ).astype(np.uint8)

    output_srgb = (image_py.linear_to_srgb(output_rgb)).astype(np.uint8)
    out_image = image_py.resize(
        Image.fromarray(output_srgb), screen.X_RES * 2, screen.Y_RES * 2,
        srgb_output=True)

    if args.show_output and canvas is not None:
        surface = pygame.surfarray.make_surface(
            np.asarray(out_image).transpose((1, 0, 2)))
        canvas.blit(surface, (0, 0))
        pygame.display.set_caption("][-Pix image preview [Iteration %d]"
                                   % seq)
        pygame.event.pump()
        pygame.display.flip()

    unique_colours = np.unique(
        palettes_rgb12_iigs.reshape(-1, 3), axis=0).shape[0]
    if args.verbose:
        print("%d unique colours" % unique_colours)

    if args.save_preview:
        if args.save_intermediate:
            outfile = "%s-%d-preview.png" % (output_base, seq)
        else:
            outfile = "%s-preview.png" % output_base
        out_image.save(outfile, "PNG")
    screen.pack()

    if args.save_intermediate:
        outfile = "%s-%d%s" % (output_base, seq, output_ext)
    else:
        outfile = "%s%s" % (output_base, output_ext)
    with open(outfile, "wb") as f:
        f.write(bytes(screen.memory))


def convert_fixed_palettes(screen, image: Image, args, fixed_scbs=False):
    """Convert image using pre-existing palettes from an SHR file."""

    from screen import SHR320Screen

    rgb = np.array(image).astype(np.float32) / 255

    base_dir = os.path.dirname(__file__)
    rgb24_to_cam16ucs = np.load(
        os.path.join(base_dir, "data/rgb24_to_cam16ucs.npy"))
    rgb12_iigs_to_cam16ucs = np.load(
        os.path.join(base_dir, "data/rgb12_iigs_to_cam16ucs.npy"))

    # Load palettes (and optionally SCBs) from the existing SHR file
    ref_file = args.palette_and_scb_file if fixed_scbs else args.palette_file
    palettes_rgb12_iigs = SHR320Screen.load_palettes(ref_file)
    fixed_line_to_palette = None
    if fixed_scbs:
        fixed_line_to_palette = SHR320Screen.load_scbs(ref_file).astype(
            np.int32)
    if args.verbose:
        print("Loaded palettes%s from %s" % (
            " and SCBs" if fixed_scbs else "", ref_file))

    # Convert palettes to CAM16UCS for perceptual dithering
    palettes_cam = np.zeros((16, 16, 3), dtype=np.float32)
    for palette_idx in range(16):
        for colour_idx in range(16):
            palettes_cam[palette_idx, colour_idx, :] = np.array(
                dither_shr_pyx.convert_rgb12_iigs_to_cam(
                    rgb12_iigs_to_cam16ucs,
                    palettes_rgb12_iigs[palette_idx, colour_idx]),
                dtype=np.float32)

    # Convert palettes to linear RGB for output rendering
    with colour.utilities.suppress_warnings(python_warnings=True):
        palettes_linear_rgb = colour.convert(
            palettes_cam, "CAM16UCS", "RGB").astype(np.float32)

    if args.show_output:
        pygame.init()
        canvas = pygame.display.set_mode((640, 400))
        canvas.fill((0, 0, 0))
        pygame.display.set_caption("][-Pix image preview")
        pygame.event.pump()
        pygame.display.flip()
    else:
        canvas = None

    colours_per_palette = 16 - getattr(args, 'reserve_colours', 0)

    # Pre-dither the image to the full 12-bit //gs palette, same as the
    # normal path, to give consistent starting pixels for the final dither.
    with colour.utilities.suppress_warnings(python_warnings=True):
        full_palette_linear_rgb = colour.convert(
            rgb12_iigs_to_cam16ucs, "CAM16UCS", "RGB").astype(np.float32)
    dither = getattr(args, 'dither', 'floyd-steinberg')
    _, image_rgb = dither_shr_pyx.dither_shr_perfect(
        rgb, rgb12_iigs_to_cam16ucs, full_palette_linear_rgb,
        rgb24_to_cam16ucs, dither)

    # Dither against the fixed palettes
    output_4bit, line_to_palette, total_image_error, _ = \
        dither_shr_pyx.dither_shr(
            image_rgb, palettes_cam, palettes_linear_rgb,
            rgb24_to_cam16ucs, colours_per_palette, dither,
            fixed_line_to_palette=fixed_line_to_palette)

    output_base, output_ext = os.path.splitext(args.output)

    _output_image(screen, output_4bit, line_to_palette, palettes_rgb12_iigs,
                  palettes_linear_rgb, args, output_base, output_ext, seq=0,
                  canvas=canvas)

    if args.show_final_score:
        print("FINAL_SCORE:", total_image_error)


def convert_one_palette(screen, image: Image, args):
    """Convert image using a single 16-colour palette for all scanlines."""

    rgb = np.array(image).astype(np.float32) / 255

    base_dir = os.path.dirname(__file__)
    rgb24_to_cam16ucs = np.load(
        os.path.join(base_dir, "data/rgb24_to_cam16ucs.npy"))
    rgb12_iigs_to_cam16ucs = np.load(
        os.path.join(base_dir, "data/rgb12_iigs_to_cam16ucs.npy"))

    reserve_colours = getattr(args, 'reserve_colours', 0)
    colours_per_palette = 16 - reserve_colours
    dither = getattr(args, 'dither', 'floyd-steinberg')

    # Convert the source image to CAM16UCS for clustering
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        image_cam = colour.convert(
            rgb, "RGB", "CAM16UCS").astype(np.float32)

    # Fit a single palette via k-means over the whole image
    pixels_cam = image_cam.reshape(-1, 3)
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=colours_per_palette, max_iter=10000)
    kmeans.fit_predict(pixels_cam)

    # Convert cluster centres to //gs 4-bit RGB
    palette_rgb12 = dither_shr_pyx.convert_cam16ucs_to_rgb12_iigs(
        kmeans.cluster_centers_.astype(np.float32))

    # Replicate into all 16 palette slots
    palettes_rgb12_iigs = np.zeros((16, 16, 3), dtype=np.uint8)
    for i in range(16):
        palettes_rgb12_iigs[i, :colours_per_palette, :] = palette_rgb12

    # Build CAM16UCS and linear-RGB palette arrays
    palettes_cam = np.zeros((16, 16, 3), dtype=np.float32)
    for pal_idx in range(16):
        for c_idx in range(16):
            palettes_cam[pal_idx, c_idx, :] = np.array(
                dither_shr_pyx.convert_rgb12_iigs_to_cam(
                    rgb12_iigs_to_cam16ucs,
                    palettes_rgb12_iigs[pal_idx, c_idx]),
                dtype=np.float32)

    with colour.utilities.suppress_warnings(python_warnings=True):
        palettes_linear_rgb = colour.convert(
            palettes_cam, "CAM16UCS", "RGB").astype(np.float32)

    if args.show_output:
        pygame.init()
        canvas = pygame.display.set_mode((640, 400))
        canvas.fill((0, 0, 0))
        pygame.display.set_caption("][-Pix image preview")
        pygame.event.pump()
        pygame.display.flip()
    else:
        canvas = None

    # Pre-dither to the full 12-bit palette
    with colour.utilities.suppress_warnings(python_warnings=True):
        full_palette_linear_rgb = colour.convert(
            rgb12_iigs_to_cam16ucs, "CAM16UCS", "RGB").astype(np.float32)
    _, image_rgb = dither_shr_pyx.dither_shr_perfect(
        rgb, rgb12_iigs_to_cam16ucs, full_palette_linear_rgb,
        rgb24_to_cam16ucs, dither)

    # Dither with all lines locked to palette 0
    fixed_line_to_palette = np.zeros(200, dtype=np.int32)
    output_4bit, line_to_palette, total_image_error, _ = \
        dither_shr_pyx.dither_shr(
            image_rgb, palettes_cam, palettes_linear_rgb,
            rgb24_to_cam16ucs, colours_per_palette, dither,
            fixed_line_to_palette=fixed_line_to_palette)

    output_base, output_ext = os.path.splitext(args.output)

    _output_image(screen, output_4bit, line_to_palette, palettes_rgb12_iigs,
                  palettes_linear_rgb, args, output_base, output_ext, seq=0,
                  canvas=canvas)

    if args.show_final_score:
        print("FINAL_SCORE:", total_image_error)


def convert(screen, image: Image, args):
    if getattr(args, 'one_palette', False):
        return convert_one_palette(screen, image, args)

    palette_and_scb_file = getattr(args, 'palette_and_scb_file', None)
    if palette_and_scb_file:
        return convert_fixed_palettes(screen, image, args,
                                      fixed_scbs=True)

    palette_file = getattr(args, 'palette_file', None)
    if palette_file:
        return convert_fixed_palettes(screen, image, args)

    rgb = np.array(image).astype(np.float32) / 255

    # Conversion matrix from RGB to CAM16UCS colour values.  Indexed by
    # 24-bit RGB value
    base_dir = os.path.dirname(__file__)
    rgb24_to_cam16ucs = np.load(
        os.path.join(base_dir, "data/rgb24_to_cam16ucs.npy"))
    rgb12_iigs_to_cam16ucs = np.load(
        os.path.join(base_dir, "data/rgb12_iigs_to_cam16ucs.npy"))

    # TODO: flags
    inner_iterations = 10
    outer_iterations = 20

    if args.show_output:
        pygame.init()
        canvas = pygame.display.set_mode((640, 400))
        canvas.fill((0, 0, 0))
        pygame.display.set_caption("][-Pix image preview")
        pygame.event.pump()  # Update caption
        pygame.display.flip()
    else:
        canvas = None

    total_image_error = None
    reserve_colours = getattr(args, 'reserve_colours', 0)
    dither = getattr(args, 'dither', 'floyd-steinberg')
    cluster_palette = ClusterPalette(
        rgb, fixed_colours=args.fixed_colours,
        reserve_colours=reserve_colours,
        dither=dither,
        rgb12_iigs_to_cam16ucs=rgb12_iigs_to_cam16ucs,
        rgb24_to_cam16ucs=rgb24_to_cam16ucs)

    output_base, output_ext = os.path.splitext(args.output)

    seq = 0
    for (
            new_total_image_error, output_4bit, line_to_palette,
            palettes_rgb12_iigs,
            palettes_linear_rgb
    ) in cluster_palette.iterate(inner_iterations, outer_iterations):

        _output_image(screen, output_4bit, line_to_palette,
                      palettes_rgb12_iigs, palettes_linear_rgb, args,
                      output_base, output_ext, seq, canvas=canvas,
                      total_image_error=total_image_error,
                      new_total_image_error=new_total_image_error)
        total_image_error = new_total_image_error

        seq += 1

    if args.show_final_score:
        print("FINAL_SCORE:", total_image_error)
