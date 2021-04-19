import base64

import numpy as np
from Crypto.Cipher import AES
from google.cloud import datastore
import re
from constants import project_mapping

# Functions for padding/unpadding according to PKCS5Padding
BLOCK_SIZE = 16
pad = lambda s: s + (BLOCK_SIZE - len(s) % BLOCK_SIZE) * chr(BLOCK_SIZE - len(s) % BLOCK_SIZE)
unpad = lambda s: s[0:-s[-1]]


def decrypt(enc, key):
    iv = enc[:16]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(enc[16:]))


def decrypt_file(filename_source, filename_destination):
    # Load data
    data = np.loadtxt(filename_source, dtype=bytes)

    # Check if this file is encrypted, i.e. contains "Encrypted" as first line
    if data[0].decode('UTF-8') != "Encrypted":
        raise Exception("File passed is not in encrypted format (doesn't contain 'Encrypted' as first line!)")

    # Extract androidID from filename
    filename_parts = re.split("[ _]+", filename_source.split("/")[-1])
    s_id = filename_parts[1]
    p_id = s_id[0:2]
    android_id = filename_parts[2]
    
    datastore_client = datastore.Client(project='specknet-pyramid-test') 

    # Get security key for android ID
    key = datastore_client.key('AndroidSecurity', android_id + "_" + p_id)    
    security_key = datastore_client.get(key)['security_key']

    # First line of decrypted file contains header
    decrypted_data = [data[1]]

    try:
        for line in data[2:]:
            try:
                string_line = line.decode('UTF-8')
                decoded_line = base64.standard_b64decode(string_line)
                decrypted_line = decrypt(decoded_line, security_key).decode('UTF-8')

                if decrypted_line[:5] != 'valid':
                    print("Invalid line detected, does not start with 'valid'! Skipping line.")
                else:
                    # decrypted data is valid
                    decrypted_data.append(decrypted_line[5:])

            except (TypeError, ValueError) as e:
                print("Skipping line, bad format: {}".format(e))

        np.savetxt(filename_destination, decrypted_data, fmt="%s")
        print("Successful decryption! Decrypted file: {}".format(filename_destination))
    except ValueError:
        raise Exception("Invalid line detected (encrypted file has been tampered with): {}".format(filename_source))


def partly_decrypt_file(filename_source, filename_destination):
    print("Partial decryption not yet implemented")
    pass

    '''
    # Load data
    data = np.loadtxt(filename_source, dtype=bytes)

    # Check if this file is encrypted, i.e. contains "Encrypted" as first line
    if data[0].decode('UTF-8') != "Encrypted":
        raise Exception("File passed is not in encrypted format (doesn't contain 'Encrypted' as first line!)")

    # Extract androidID from filename
    s_id = filename_source.split("/")[-1].split(' ')[1]
    p_id = s_id[0:2]
    android_id = filename_source.split("/")[-1].split(' ')[2]

    datastore_client = datastore.Client()

    # Get security key for android ID
    key = datastore_client.key('AndroidSecurity', android_id + "_" + p_id)
    security_key = datastore_client.get(key)['security_key']

    # First line of decrypted file contains header
    decrypted_data = [data[1]]

    try:
        for line in data[2:]:
            try:
                string_line = line.decode('UTF-8')
                decoded_line = base64.standard_b64decode(string_line)
                decrypted_line = decrypt(decoded_line, security_key).decode('UTF-8')

                if decrypted_line[:5] != 'valid':
                    print("Invalid line detected, does not start with 'valid'! Skipping line.")
                else:
                    # decrypted data is valid
                    decrypted_data.append(decrypted_line[5:])
            except (TypeError, ValueError) as e:
                print("Skipping line, bad format.")

        np.savetxt(filename_destination, decrypted_data, fmt="%s")
        print("Successful decryption! Decrypted file: {}".format(filename_destination))
    except ValueError:
        raise Exception("Invalid line detected (encrypted file has been tampered with): {}".format(filename_source))
    '''