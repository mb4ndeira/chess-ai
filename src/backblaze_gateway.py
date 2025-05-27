from b2sdk.v2 import InMemoryAccountInfo, B2Api

class BackblazeGateway:
    def __init__(self, application_key_id, application_key, bucket_name):
        self.info = InMemoryAccountInfo()
        self.b2_api = B2Api(self.info)
        self.b2_api.authorize_account("production", application_key_id, application_key)
        self.bucket = self.b2_api.get_bucket_by_name(bucket_name)

    def upload_file(self, local_path, remote_path, file_info=None):
        if file_info is None:
            file_info = {"source": "ml-model"}
        self.bucket.upload_local_file(
            local_file=local_path,
            file_name=remote_path,
            file_infos=file_info,
        )