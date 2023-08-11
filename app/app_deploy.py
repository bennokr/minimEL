from flask import Flask, request, abort
import hmac
import hashlib
import logging
import os
import git

w_secret = os.environ.get('SECRET')
logging.error(w_secret)

def is_valid_signature(x_hub_signature, data, private_key):
    # x_hub_signature and data are from the webhook payload
    # private key is your webhook secret
    hash_algorithm, github_signature = x_hub_signature.split('=', 1)
    algorithm = hashlib.__dict__.get(hash_algorithm)
    encoded_key = bytes(private_key, 'latin-1')
    mac = hmac.new(encoded_key, msg=data, digestmod=algorithm)
    return hmac.compare_digest(mac.hexdigest(), github_signature)


def update():
    # From https://github.com/SwagLyrics/swaglyrics-backend
    if request.method != 'POST':
        return 'OK'
    else:
        abort_code = 418
        if request.headers.get('X-GitHub-Event') != "push":
            abort(abort_code)
        x_hub_signature = request.headers.get('X-Hub-Signature')
        if not is_valid_signature(x_hub_signature, request.data, w_secret):
            logging.warning('Deploy signature failed: {sig}'.format(sig=x_hub_signature))
            abort(abort_code)
        if request.get_json().get('ref') != 'refs/heads/master':
            logging.warning('Deploy request for wrong ref')
            abort(abort_code)
        
        basedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        repo = git.Repo(basedir)
        origin = repo.remotes.origin
        pull_info = origin.pull()
        commit_hash = pull_info[0].commit.hexsha
        return 'Updated server to commit {commit}'.format(commit=commit_hash)
