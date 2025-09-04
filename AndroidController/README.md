# AndroidController

## Sending messages to the Android App

### Display info under main page
`{"cat": "info", "value": "hello"}`: Displays information on the main page under, "Robot Status".

### Display car position
`{"cat": "location", "value": {"x": "0", "y": "10", "d": "2"}}` Displays the car position on the map

### Update car progress
`{"cat": "status", "value": "running"}`: Tells the android app that the car is still running. We will be no grid interaction in the app
`{"cat": "status", "value": "finished"}`: Tells the android app that the car is complete and displays the time taken. Returns full grid interaction

### Updating car mode
`{"cat": "mode", "value": "manual"}` and `{"cat": "mode", "value": "path"}`: Sends information to the RPi regarding the mode of the car

### Updating the obstacle information after successful image recognition
`{"cat": "image-rec", "value": {"obstacle_id": "1", "image_id": "A"}}`: `obstacle_id` is the obstacle number we see on the android app. `image_id` is the actual image recognised by the model. Updates the  obstacle with `obstacle_id` on the android app to the image recognised by the model, `image_id`.
