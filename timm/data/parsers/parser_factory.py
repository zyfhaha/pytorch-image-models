import os

from .parser_image_folder import ParserImageFolder
from .parser_image_tar import ParserImageTar
from .parser_image_in_tar import ParserImageInTar
from .parser_image_json import ParserImageJson


def create_parser(name, root, split='train', json_file=None, use_cache=False, **kwargs):
    name = name.lower()
    name = name.split('/', 2)
    prefix = ''
    if len(name) > 1:
        prefix = name[0]
    name = name[-1]
    # FIXME improve the selection right now just tfds prefix or fallback path, will need options to
    # explicitly select other options shortly
    if prefix == 'tfds':
        from .parser_tfds import ParserTfds  # defer tensorflow import
        parser = ParserTfds(root, name, split=split, shuffle=kwargs.pop('shuffle', False), **kwargs)
    else:
        assert os.path.exists(root)
        # default fallback path (backwards compat), use image tar if root is a .tar file, otherwise image folder
        # FIXME support split here, in parser?
        if os.path.isfile(root) and os.path.splitext(root)[1] == '.tar':
            parser = ParserImageInTar(root, **kwargs)
        elif json_file is not None:
            print("validation use json parser")
            parser = ParserImageJson(root, json_file, use_cache=use_cache)
        else:
            parser = ParserImageFolder(root, use_cache=use_cache, **kwargs)
    return parser
