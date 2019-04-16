import urllib
import tarfile
import errno
import os

def _run_download(args):
    try:
        os.makedirs(args.directory)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    if args.selection:
        filepath = args.directory + '/transitions_selection.tar.gz'
        url = 'https://berkeley.box.com/shared/static/mk7gjd881rn6k710h4aqnx8ywei9hgvl.gz'
    else:
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
