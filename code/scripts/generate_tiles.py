import sys
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from datetime import datetime

import click
import rasterio as rio
from rasterio.windows import Window

from tiling.const_stride import ConstStrideTiles
from tiling.const_size import ConstSizeTiles

from dataflow.io_utils import imwrite_rasterio, read_data_rasterio


def get_files_from_folder(input_dir, extensions=None):
    """Method to get files from a folder with optional filter on extensions
    Args:
      input_dir: input folder
      extensions (list or tuple): List of extensions to filter files (Default value = None)

    Returns:
      List of filepaths
    """
    output = []
    if extensions is None:
        extensions = [""]
    elif not isinstance(extensions, (tuple, list)):
        extensions = [extensions]

    for ext in extensions:
        files = Path(input_dir).rglob("*{}".format(ext))
        output.extend([f.as_posix() for f in files if f.is_file()])

    return output


@click.group()
def cli():
    pass


def _parse_options(options):
    # this import is needed to correctly run `eval`
    from rasterio.enums import Resampling
    assert isinstance(options, str), "Options should be a string"
    output = {}
    if len(options) == 0:
        return output
    options = options.split(';')
    for opt in options:
        assert "=" in opt, "Option '{}' should contain '='".format(opt)
        k, v = opt.split('=')
        output[k] = eval(v)
    return output


def run_task(filepath, output_dir, get_tiles_fn, output_extension, options):
    try:
        src = rio.open(filepath)
    except rio.errors.RasterioIOError as e:
        raise RuntimeError("Failed to open file: '%s'. Check if it exists or has supported format." % filepath +
                           "\nRasterio error message: {}".format(e))

    output_tiles_dir = Path(output_dir) / "{}_tiles".format(Path(filepath).stem)
    output_tiles_dir.mkdir(exist_ok=True)

    tiles = get_tiles_fn((src.width, src.height))

    output_extension = Path(filepath).suffix[1:] if output_extension is None else output_extension
    kwargs = {}
    for extent, out_size in tiles:

        x, y, w, h = extent
        # get data
        tile = read_data_rasterio(src, src_rect=[x, y, w, h],
                                  dst_width=out_size[0],
                                  dst_height=out_size[1],
                                  **_parse_options(options))

        kwargs['crs'] = src.crs
        if src.transform is not None:
            kwargs['transform'] = src.window_transform(Window(x, y, w, h))

        output_tile_filepath = output_tiles_dir / ("tile_%i_%i.%s" % (x, y, output_extension))
        imwrite_rasterio(output_tile_filepath.as_posix(), tile, **kwargs)

    src.close()


def _run_xyz_tiler(get_tiles_fn, input_dir_or_file, output_dir, extensions, output_extension,
                   n_workers, options, without_log_file, quiet, conf_str):

    if not quiet:
        click.echo(conf_str)

    if Path(input_dir_or_file).is_dir():
        if extensions is not None:
            extensions = extensions.split(",")
        files = get_files_from_folder(input_dir_or_file, extensions)
        assert len(files) > 0, "No files with extensions '{}' found at '{}'".format(extensions, input_dir_or_file)
    else:
        files = [input_dir_or_file]

    if not Path(output_dir).exists():
        if not quiet:
            click.echo("Create output folder: %s" % output_dir)
        Path(output_dir).mkdir(parents=True)

    if not without_log_file:
        cmd = sys.argv
        now = datetime.now()
        log_filepath = Path(output_dir) / ("%s.log" % now.strftime("%Y%m%d_%H%M%S"))
        with log_filepath.open('w') as handler:
            handler.write("Command:\n")
            cmd_str = " ".join(cmd)
            handler.write(cmd_str + "\n\n")
            handler.write(conf_str + "\n")

    func = partial(run_task,
                   output_dir=output_dir,
                   get_tiles_fn=get_tiles_fn,
                   output_extension=output_extension,
                   options=options)

    progressbar = click.progressbar if not quiet else EmptyBar

    chunk_size = 10
    if n_workers > 1 and len(files) > chunk_size // 2:
        with Pool(n_workers) as pool:
            with progressbar(length=len(files)) as bar:
                for i in range(0, len(files), chunk_size):
                    chunk_files = files[i: i + chunk_size]
                    pool.map(func, chunk_files)
                    bar.update(chunk_size)
    else:
        with progressbar(files, label='Run tile generator on files') as bar:
            for f in bar:
                func(f)


@click.command()
@click.argument('input_dir_or_file', type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.argument('output_dir', type=click.Path())
@click.argument('tile_size', nargs=2, type=int)
@click.argument('stride', nargs=2, type=int)
@click.option('--origin', nargs=2, type=int, default=(0, 0),
              help="Point in pixels in the original image from where to start the tiling. " +
                   "Values can be positive or negative")
@click.option('--scale', type=float, default=1.0,
              help="Scaling applied to the input image parameters before extracting tile's extent." +
                   "For example, scale of 0.75 corresponds to a zoom out")
@click.option('--without_nodata', type=bool, is_flag=True,
              help="Do not include nodata. Default, nodata is included. If nodata is included then tile extents " +
                   "have all the same size, otherwise tiles at boundaries will be reduced.")
@click.option('--extensions', type=str, default=None,
              help="String of file extensions to select (if input is a directory), e.g. 'jpg,png,tif'")
@click.option('--output_extension', type=str, default=None, help="Output tile file extension. " +
                                                                 "By default, input file extension is taken")
@click.option('--n_workers', default=4, type=int, help="Number of workers in the processing pool [default=4]")
@click.option('--options', type=str, default="", help="Options to pass when read data with rasterio. " +
                                                      "Example --options='resampling=Resampling.nearest;" +
                                                      "dtype=np.float32;nodata_value=-1'")
@click.option('-q', '--quiet', is_flag=True, help='Disable verbose mode')
@click.option('--without_log_file', type=bool, is_flag=True,
              help="Do not write a log file in the output folder")
def run_const_stride_tiler(input_dir_or_file, output_dir, tile_size, stride, origin, scale, without_nodata, extensions,
                           output_extension, n_workers, options, quiet, without_log_file):

    conf_str = """
        input: {}
        output: {}
        tile size: {}
        stride: {}
        origin: {}
        scale: {}
        without_nodata: {}
        extensions: {}
        output_ext: {}
        n_workers: {}
        options: {}
        without_log_file: {}
    """.format(input_dir_or_file, output_dir, tile_size, stride, origin, scale, without_nodata, extensions,
               output_extension, n_workers, options, without_log_file)

    get_tiles_fn = partial(ConstStrideTiles, tile_size=tile_size, stride=stride, scale=scale, origin=origin,
                           include_nodata=not without_nodata)

    _run_xyz_tiler(get_tiles_fn, input_dir_or_file, output_dir, extensions, output_extension,
                   n_workers, options, without_log_file, quiet, conf_str)


@click.command()
@click.argument('input_dir_or_file', type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.argument('output_dir', type=click.Path())
@click.argument('tile_size', nargs=2, type=int)
@click.argument('min_overlapping', type=int)
@click.option('--scale', type=float, default=1.0,
              help="Scaling applied to the input image parameters before extracting tile's extent." +
                   "For example, scale of 0.75 corresponds to a zoom out")
@click.option('--extensions', type=str, default=None,
              help="String of file extensions to select (if input is a directory), e.g. 'jpg,png,tif'")
@click.option('--output_extension', type=str, default=None, help="Output tile file extension. " +
                                                                 "By default, input file extension is taken")
@click.option('--n_workers', default=4, type=int, help="Number of workers in the processing pool [default=4]")
@click.option('--options', type=str, default="", help="Options to pass when read data with rasterio. " +
                                                      "Example --options='resampling=Resampling.nearest;" +
                                                      "dtype=np.float32;nodata_value=-1'")
@click.option('-q', '--quiet', is_flag=True, help='Disable verbose mode')
@click.option('--without_log_file', type=bool, is_flag=True,
              help="Do not write a log file in the output folder")
def run_const_size_tiler(input_dir_or_file, output_dir, tile_size, min_overlapping, scale, extensions,
                         output_extension, n_workers, options, quiet, without_log_file):

    conf_str = """
        input: {}
        output: {}
        tile size: {}
        min_overlapping: {}
        scale: {}
        extensions: {}
        output_ext: {}
        n_workers: {}
        options: {}
        without_log_file: {}
    """.format(input_dir_or_file, output_dir, tile_size, min_overlapping, scale, extensions,
               output_extension, n_workers, options, without_log_file)

    get_tiles_fn = partial(ConstSizeTiles, tile_size=tile_size, min_overlapping=min_overlapping, scale=scale)

    _run_xyz_tiler(get_tiles_fn, input_dir_or_file, output_dir, extensions, output_extension,
                   n_workers, options, without_log_file, quiet, conf_str)


class EmptyBar(object):
    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable

    def __enter__(self):
        return self.iterable

    def __exit__(self, *args, **kwargs):
        pass

    def update(self, n_steps):
        pass


cli.add_command(run_const_stride_tiler, name="const_stride")
cli.add_command(run_const_size_tiler, name="const_size")


if __name__ == "__main__":
    cli()
