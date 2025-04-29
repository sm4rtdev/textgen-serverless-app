This is the back end code for a serverless text generation app to be deployed on Google Cloud Functions. 

My set up took 2GB memory to work properly. 

Once your model is deployed, you can use the following command to test the function:

`curl -XPOST https://us-central1-URL-FROM-GCP_UNDER_TRIGGER -H 'Content-Type: application/json' -d @test.json`
