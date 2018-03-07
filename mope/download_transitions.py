import urllib
import tarfile
import os
import errno

def _run_download(args):
    try:
        os.makedirs(args.directory)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    filepath = args.directory + '/transitions.tar.gz'
    url = 'https://berkeley.box.com/shared/static/27ghsfp00xa7g8470ndrp5ft49y7a47y.gz'
    urllib.urlretrieve(url, filepath)

    try:
        tar = tarfile.open(filepath, "r:gz")
        tar.extractall()
        tar.close()
    except:
        raise OSError('Could not extract transitions file')

    try:
        os.remove(filepath)
    except OSError:
        raise OSError('Could not delete transitions tarfile')
