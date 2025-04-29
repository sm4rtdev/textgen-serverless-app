This is the back end code for a serverless text generation app to be deployed on Google Cloud Functions. 

You can use the contents from these files to follow along the instructions [here](https://medium.com/swlh/deploying-pytorch-models-to-serverless-environments-8fcd25dc7b5c) to set up the app. 

My set up took 2GB memory to work properly. 

Once your model is deployed, you can use the following command to test the function:

`curl -XPOST https://us-central1-URL-FROM-GCP_UNDER_TRIGGER -H 'Content-Type: application/json' -d @test.json`
