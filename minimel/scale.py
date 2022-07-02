import os
import re
import codecs
import time
import logging
import dask.distributed


def progress(*args, **kwargs):
    start = time.time()
    p = dask.distributed.progress(*args, **kwargs)
    end = time.time()
    logging.info(f'Finished in {int(end - start)}s')
    return p


def get_client(*args, **kwargs):
    try:
        return dask.distributed.get_client(*args, **kwargs)
    except:
        return dask.distributed.Client(*args, **kwargs)


def fileparts(ns, path, nparts, start, end, chunksize=2**12):
    parts = []
    utf8_decoder = codecs.getincrementaldecoder("utf8")()
    for n in ns:
        loc, size = 0, os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(int(size * (n / nparts)))
            part = ""
            for chunk in iter(lambda: f.read(chunksize), b""):
                while chunk:
                    try:
                        chunk = utf8_decoder.decode(chunk)
                    except:
                        chunk = chunk[1:]
                    else:
                        break
                loc += chunksize
                part += chunk
                while (start in part) and (end in part[part.index(start) :]):
                    s = part.index(start)
                    e = s + part[s:].index(end) + len(end)
                    parts.append(part[s:e])
                    part = part[e:]
                if loc > (size / nparts):
                    break
    return parts
