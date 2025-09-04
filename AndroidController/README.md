# AndroidController

## Robot Directions
- 0: Up
- 2: Right
- 4: Down
- 6: Left

## Sending messages to the Android App

### Display info under main page
`{"cat": "info", "value": "hello"}`: Displays information on the main page under, "Robot Status".

### Display car position
`{"cat": "location", "value": {"x": "0", "y": "10", "d": "2"}}` Displays the car position on the map.

**Car must be placed onto the map first for the change of direction to draw properly. Else only the position will be drawn**

### Update car progress
`{"cat": "status", "value": "running"}`: Tells the android app that the car is still running. We will be no grid interaction in the app
`{"cat": "status", "value": "finished"}`: Tells the android app that the car is complete and displays the time taken. Returns full grid interaction

### Updating car mode
`{"cat": "mode", "value": "manual"}` and `{"cat": "mode", "value": "path"}`: Sends information to the RPi regarding the mode of the car

### Updating the obstacle information after successful image recognition
`{"cat": "image-rec", "value": {"obstacle_id": "1", "image_id": "A"}}`: `obstacle_id` is the obstacle number we see on the android app. `image_id` is the actual image recognised by the model. 

I.e. Updates the  obstacle with `obstacle_id` on the android app to the image recognised by the model, `image_id`.

## Key Functions
These functions, in `HomeFragment.java`, receive the function inputs from `BluetoothConnnectionService.java`.

### `roboStatusUpdateReceiver`

### `roboStateReceiver`

### `roboModeUpdateReceiver`

### `updateObstalceListReceiver`

### `imageRecResultReceiver`

### `imageRecResultReceiver`

