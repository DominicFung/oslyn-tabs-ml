# aws s3 sync --storage-class REDUCED_REDUNDANCY --delete --exact-timestamps . s3://oslyn-tabs-ml --profile a1
python3 s3sync.py --config config.yml push --s3path s3://oslyn-tabs-ml --localpath .