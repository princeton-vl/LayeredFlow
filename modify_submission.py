import requests
import argparse
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor
from tqdm import tqdm
import os

website = 'https://layeredflow.cs.princeton.edu'

session = requests.Session()    
response = session.get(f'{website}/request_submit/')
csrf_token = session.cookies.get('csrftoken')
headers = {
    'X-CSRFToken': csrf_token,
    'Referer': f'{website}/request_submit/'
}

def request_verification(args, url=f"{website}/request_modify"):
    response = session.post(f"{url}/{args.id}/", data={'email': args.email}, headers=headers)

    if response.status_code == 200:
        upload_id = response.json()['upload_id']
        print(f"Verification code sent to {args.email}.")
        print(response.json()['message'])
        return upload_id
    else:
        print("Failed to request verification:", response.text)
        exit(1)

def verify_code(upload_id, code, url=f"{website}/verify"):
    data = {
        'code': code,
    }
    response = session.post(f"{url}/{upload_id}/", data=data, headers=headers)
    if response.status_code == 200:
        return True
    else:
        print(response)
        exit(1)

def modify_submission(upload_id,  url=f"{website}/modify"):
    data = {
        'method_name': args.method_name,
        'publication': args.publication,
        'anonymous': args.anonymous,
    }
    if args.url_publication:
        data['url_publication'] = args.url_publication
    if args.url_code:
        data['url_code'] = args.url

    response = session.post(f"{url}/{upload_id}/", data=data, headers=headers)
    if response.status_code == 200:
        print('Submission modified successfully.')
        print('If you make anonymous=False, your submission will be displayed in the leaderboard.')
        return True
    else:
        print(response)
        exit(1)
  

def compress_folder(path):
    tmp_folder = os.path.join(os.path.dirname(path), 'tmp')
    os.makedirs(tmp_folder, exist_ok=True)
    os.system(f"tar -cvzf - {path} | split -a 4 -b 512M -d - {tmp_folder}/submission.tar.gz.")
    return tmp_folder

def main(args):
    upload_id = request_verification(args)
    if upload_id:
        code = input("Please enter the verification code sent to your email: ")

        if verify_code(upload_id, code):
            modify_submission(upload_id)
        else:
            print("Verification failed. Upload not allowed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="Upload ID", required=True)
    parser.add_argument("--email", help="Email address to send verification code to", required=True)
    parser.add_argument("--method_name", help="Method name that will be displayed in the leaderboard", required=True)
    parser.add_argument("--publication", help="Publication name", default="Anonymous")
    parser.add_argument("--url_publication", help="Publication URL", default=None)
    parser.add_argument("--url_code", help="Code URL", default=None)
    parser.add_argument("--anonymous", help="Whether to be anonymous", default=False)
                        
    args = parser.parse_args()
    main(args)
