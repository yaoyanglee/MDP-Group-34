package com.example.androidcontroller;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.widget.SwitchCompat;
import androidx.fragment.app.Fragment;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import android.os.Handler;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ListAdapter;
import android.widget.ListView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import android.content.SharedPreferences;

import org.json.JSONArray;
import org.json.JSONObject;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Timer;
import java.util.TimerTask;

public class HomeFragment extends Fragment{
    public static String TAG = "HomeFragment";

    private boolean initializedIntentListeners = false;
    private TextView txtRoboStatus;

    private Switch manualModeSwitch;
    private Switch outdoorArenaSwitch;
    private Switch turningModeSwitch;
    private SwitchCompat themeSwitch;

    private SharedPreferences sharedPreferences;

    private View rootview;

    //For Arena
    boolean placingRobot, settingObstacle, settingDir;

    private Handler handler = new Handler();

    //GridMap
    private GridMap gridMap;

    //For robot
    private boolean isManual = false;

    //For Obstalce listview
    private ObstaclesListViewAdapter obstaclesListViewAdapter;
    private List<ObstacleListItem> obstacleListItemList;
    private ListView obstacleListView;

    //Auxiliary
    private long timeStarted;
    private long timeEnded;
    private long timeTakenInNanoSeconds;

    //Android widgets for UI
    //ROBOT RELATED
    Button btnSendArenaInfo;
    Button btnSendStartImageRec;
    Button btnSendStartFastestCar;

    //ARENA RELATED
    Button btnResetArena;
    Button btnSetObstacle;
    Button btnSetFacing;
    Button btnPlaceRobot;

    //Adding obstacles using buttons
    Button btnAddObsManual;
    EditText addObs_x;
    EditText addObs_y;

    //Bot Status
    TextView txtTimeTaken;

    // TODO: Rename parameter arguments, choose names that match
    // the fragment initialization parameters, e.g. ARG_ITEM_NUMBER
    private static final String ARG_PARAM1 = "param1";
    private static final String ARG_PARAM2 = "param2";

    // TODO: Rename and change types of parameters
    private String mParam1;
    private String mParam2;

    public HomeFragment() {
        // Required empty public constructor
    }

    @Override
    public void onAttach(@NonNull Context context) {
        super.onAttach(context);
        sharedPreferences = context.getSharedPreferences(RobotControllerActions.PREFS_NAME, Context.MODE_PRIVATE);
    }

    /**
     * Use this factory method to create a new instance of
     * this fragment using the provided parameters.
     *
     * @param param1 Parameter 1.
     * @param param2 Parameter 2.
     * @return A new instance of fragment ArenaFragment.
     */
    // TODO: Rename and change types and number of parameters
    public static HomeFragment newInstance(String param1, String param2) {
        HomeFragment fragment = new HomeFragment();
        Bundle args = new Bundle();
        args.putString(ARG_PARAM1, param1);
        args.putString(ARG_PARAM2, param2);
        fragment.setArguments(args);
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        obstacleListItemList = new ArrayList<>();

        if (getArguments() != null) {
            mParam1 = getArguments().getString(ARG_PARAM1);
            mParam2 = getArguments().getString(ARG_PARAM2);
        }

        if(!initializedIntentListeners){
            LocalBroadcastManager.getInstance(getContext()).registerReceiver(roboStatusUpdateReceiver, new IntentFilter("updateRobocarStatus"));
            LocalBroadcastManager.getInstance(getContext()).registerReceiver(roboStateReceiver, new IntentFilter("updateRoboCarState"));
            LocalBroadcastManager.getInstance(getContext()).registerReceiver(roboModeUpdateReceiver, new IntentFilter("updateRobocarMode"));
            LocalBroadcastManager.getInstance(getContext()).registerReceiver(updateObstalceListReceiver, new IntentFilter("newObstacleList"));
            LocalBroadcastManager.getInstance(getContext()).registerReceiver(imageRecResultReceiver, new IntentFilter("imageResult"));
            LocalBroadcastManager.getInstance(getContext()).registerReceiver(robotLocationUpdateReceiver, new IntentFilter("updateRobocarLocation"));
            LocalBroadcastManager.getInstance(getContext()).registerReceiver(startFastestCarReceiver, new IntentFilter(RobotControllerActions.ACTION_START_FASTEST_CAR));

            initializedIntentListeners = true;
        }
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {

        rootview = inflater.inflate(R.layout.fragment_home, container, false);

        gridMap = rootview.findViewById(R.id.mapView);
        if (gridMap == null) {
            throw new IllegalStateException("Map view not found in layout");
        }

        //For obstacle list view
        obstacleListView = rootview.findViewById(R.id.home_obstacles_listview);
        obstaclesListViewAdapter = new ObstaclesListViewAdapter(getContext(), R.layout.home_obstacle_list_layout, obstacleListItemList);
        obstacleListView.setAdapter(obstaclesListViewAdapter);
        refreshObstacleListHeight();

        //Theme toggle
        themeSwitch = rootview.findViewById(R.id.switch_theme);
        if (themeSwitch != null) {
            boolean isDarkMode = ThemeUtils.isDarkModeEnabled(requireContext());
            themeSwitch.setChecked(isDarkMode);
            themeSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> ThemeUtils.setDarkMode(requireContext(), isChecked));
        }

        //Switches
        manualModeSwitch = (Switch) rootview.findViewById(R.id.switch_manualMode);
        outdoorArenaSwitch = (Switch) rootview.findViewById(R.id.switch_outdoor);
        turningModeSwitch = (Switch) rootview.findViewById(R.id.switch_turnmode);

        if(sharedPreferences == null && getContext() != null){
            sharedPreferences = requireContext().getSharedPreferences(RobotControllerActions.PREFS_NAME, Context.MODE_PRIVATE);
        }

        boolean savedOutdoorArena = false;
        boolean savedBigTurn = false;
        if(sharedPreferences != null){
            savedOutdoorArena = sharedPreferences.getBoolean(RobotControllerActions.PREF_OUTDOOR, false);
            savedBigTurn = sharedPreferences.getBoolean(RobotControllerActions.PREF_BIG_TURN, false);
        }

        outdoorArenaSwitch.setChecked(savedOutdoorArena);
        turningModeSwitch.setChecked(savedBigTurn);
        gridMap.setIsOutdoorArena(savedOutdoorArena);

        manualModeSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if(isChecked){
                    sendModeCmdIntent("manual");
                }else{
                    sendModeCmdIntent("path");
                }
            }
        });

        outdoorArenaSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (gridMap != null) {
                    gridMap.setIsOutdoorArena(isChecked);
                }
                if(sharedPreferences != null){
                    sharedPreferences.edit().putBoolean(RobotControllerActions.PREF_OUTDOOR, isChecked).apply();
                }
            }
        });

        turningModeSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if(sharedPreferences != null){
                    sharedPreferences.edit().putBoolean(RobotControllerActions.PREF_BIG_TURN, isChecked).apply();
                }
            }
        });

        //Initialize Flags
        placingRobot = false;

        // For updating of robot status
        this.txtRoboStatus = (TextView) rootview.findViewById(R.id.robotStatusText);

        //CONTROL BUTTON DECLARATIONS
        ImageButton controlBtnUp = rootview.findViewById(R.id.upArrowBtn);
        ImageButton controlBtnDown = rootview.findViewById(R.id.downArrowBtn);
        ImageButton controlBtnLeft = rootview.findViewById(R.id.leftArrowBtn);
        ImageButton controlBtnRight = rootview.findViewById(R.id.rightArrowBtn);

        attachManualControlListener(controlBtnUp, "FW--");
        attachManualControlListener(controlBtnDown, "BW--");
        attachManualControlListener(controlBtnLeft, "TL--");
        attachManualControlListener(controlBtnRight, "TR--");

        //TIME TAKEN TEXTVIEW
        txtTimeTaken = rootview.findViewById(R.id.txt_timeTaken);

        //ROBOT RELATED
        btnSendArenaInfo = rootview.findViewById(R.id.btnSendInfo);
        btnSendStartImageRec = rootview.findViewById(R.id.btnStartImageRec);
        btnSendStartFastestCar = rootview.findViewById(R.id.btnStartFastestCar);

        //ARENA RELATED
        btnResetArena = rootview.findViewById(R.id.btnResetArena);
        btnSetObstacle = rootview.findViewById(R.id.btnSetObstacle);
        btnSetFacing = rootview.findViewById(R.id.btnDirectionFacing);
        btnPlaceRobot = rootview.findViewById(R.id.btnPlaceRobot);

        //Adding obstacles using buttons
        btnAddObsManual = rootview.findViewById(R.id.add_obs_btn);
        addObs_x = rootview.findViewById(R.id.add_obs_x_value);
        addObs_y = rootview.findViewById(R.id.add_obs_y_value);

        // OnClickListeners for sending arena info to RPI
        btnSendArenaInfo.setOnClickListener(v->{
            if (gridMap != null) {
                gridMap.sendUpdatedObstacleInformation();
            }
        });

        btnSendStartImageRec.setOnClickListener(v->{
            if (gridMap != null) {
                gridMap.removeAllTargetIDs();
            }
            txtTimeTaken.setVisibility(View.INVISIBLE);
            sendControlCmdIntent("start");
            timeStarted = System.nanoTime();
            new Timer().schedule(new TimerTask() {
                @Override
                public void run() {
                    sendControlCmdIntent("stop");
                }
            }, 360000);
        });

        btnSendStartFastestCar.setOnClickListener(v->{
            boolean isBigTurn = turningModeSwitch != null && turningModeSwitch.isChecked();
            boolean isOutdoor = outdoorArenaSwitch != null && outdoorArenaSwitch.isChecked();
            startFastestCar(isBigTurn, isOutdoor);
        });

        btnResetArena.setOnClickListener(v->{
            try{
                if (gridMap != null) {
                    gridMap.resetMap();
                }
            }catch (Exception e){
                Log.e(TAG, "onCreateView: An error occured while resetting map");
                e.printStackTrace();
            }
        });

        // OnClickListeners for the arena related buttons
        btnPlaceRobot.setOnClickListener(v -> {
            try{
                //New status
                placingRobot = !placingRobot;
                if(placingRobot){
                    if (gridMap != null) {
                        gridMap.setStartCoordStatus(placingRobot);
                    }
                    btnPlaceRobot.setText("Stop Set Robot");

                    //Disable other buttons
                    btnSetObstacle.setEnabled(false);
                    btnSetFacing.setEnabled(false);
                    btnResetArena.setEnabled(false);
                    btnSendStartFastestCar.setEnabled(false);
                    btnSendStartImageRec.setEnabled(false);
                }else{
                    if (gridMap != null) {
                        gridMap.setStartCoordStatus(placingRobot);
                    }
                    btnSetObstacle.setEnabled(true);
                    btnSetFacing.setEnabled(true);
                    btnResetArena.setEnabled(true);
                    btnSendStartFastestCar.setEnabled(true);
                    btnSendStartImageRec.setEnabled(true);
                    btnPlaceRobot.setText("Place Robot");
                }
            }catch (Exception e){
                Log.e(TAG, "onCreateView: An error occured while placing robot");
                e.printStackTrace();
            }
        });

        btnSetObstacle.setOnClickListener(v->{
            try{
                settingObstacle = !settingObstacle;
                if(settingObstacle){
                    if (gridMap != null) {
                        gridMap.setSetObstacleStatus(settingObstacle);
                    }
                    btnSetObstacle.setText("Stop Set Obstacle");

                    //Disable other buttons
                    btnSetFacing.setEnabled(false);
                    btnPlaceRobot.setEnabled(false);
                    btnResetArena.setEnabled(false);
                    btnSendStartFastestCar.setEnabled(false);
                    btnSendStartImageRec.setEnabled(false);
                }else{
                    if (gridMap != null) {
                        gridMap.setSetObstacleStatus(settingObstacle);
                    }
                    btnSetObstacle.setText("Set Obstacle");

                    //Re-enable other buttons
                    btnSetFacing.setEnabled(true);
                    btnPlaceRobot.setEnabled(true);
                    btnResetArena.setEnabled(true);
                    btnSendStartFastestCar.setEnabled(true);
                    btnSendStartImageRec.setEnabled(true);
                }
            }catch (Exception e){
                Log.e(TAG, "onCreateView: An error occurred while setting obstacle");
                e.printStackTrace();
            }
        });

        btnSetFacing.setOnClickListener(v -> {
            try{
                settingDir = !settingDir;
                if(settingDir){
                    gridMap.setSetDirection(settingDir);
                    btnSetFacing.setText("Stop Set Facing");

                    //Disable Other Buttons
                    btnSetObstacle.setEnabled(false);
                    btnPlaceRobot.setEnabled(false);
                    btnResetArena.setEnabled(false);
                    btnSendStartFastestCar.setEnabled(false);
                    btnSendStartImageRec.setEnabled(false);
                }else{
                    if (gridMap != null) {
                        gridMap.setSetDirection(settingDir);
                    }
                    btnSetFacing.setText("Set Facing");

                    //Reenable other buttons
                    btnSetObstacle.setEnabled(true);
                    btnPlaceRobot.setEnabled(true);
                    btnResetArena.setEnabled(true);
                    btnSendStartFastestCar.setEnabled(true);
                    btnSendStartImageRec.setEnabled(true);
                }
            }catch (Exception e){
                Log.e(TAG, "onCreateView: An error occurred while setting obstacle direction");
                e.printStackTrace();
            }
        });

        btnAddObsManual.setOnClickListener(v -> {
            try{
                String x_value = addObs_x.getText().toString();
                String y_value = addObs_y.getText().toString();
                try
                {
                    int x_value_int = Integer.parseInt(x_value);
                    int y_value_int = Integer.parseInt(y_value);

                    if( x_value_int < 20 && x_value_int >=0 && y_value_int < 20 && y_value_int >=0){
                        if (gridMap != null) {
                            gridMap.setObstacleCoord(x_value_int, y_value_int);
                        }
                        showShortToast("Added obstacle");
                        addObs_x.setText("");
                        addObs_y.setText("");
                    }else{
                        showShortToast("Invalid Coordinates");
                    }
                }catch (Exception e){
                    showShortToast("Incorrect values!");
                }
            }catch (Exception e){
                Log.e(TAG, "onCreateView: An error occurred while adding obstacle manually");
                e.printStackTrace();
            }
        });

        // DEBUGGING BUTTONS
        /*
        Button btnFW10 = rootview.findViewById(R.id.temp_btnFW10);
        btnFW10.setOnClickListener(v -> {sendDirectionCmdIntent("FW10");});
        Button btnBT10 = rootview.findViewById(R.id.temp_btnBW10);
        btnBT10.setOnClickListener(v -> {sendDirectionCmdIntent("BW10");});
        Button btnFL00 = rootview.findViewById(R.id.temp_btnFL00);
        btnFL00.setOnClickListener(v -> {sendDirectionCmdIntent("FL00");});
        Button btnFR00 = rootview.findViewById(R.id.temp_btnFR00);
        btnFR00.setOnClickListener(v -> {sendDirectionCmdIntent("FR00");});
        Button btnBL00 = rootview.findViewById(R.id.temp_btnBL00);
        btnBL00.setOnClickListener(v -> {sendDirectionCmdIntent("BL00");});
        Button btnBR00 = rootview.findViewById(R.id.temp_btnBR00);
        btnBR00.setOnClickListener(v->{sendDirectionCmdIntent("BR00");});
         */
        // Inflate the layout for this fragment
        return rootview;
    }
    @Override
    public void onDestroyView() {
        super.onDestroyView();
        gridMap = null;
    }

    private void startFastestCar(boolean isBigTurn, boolean isOutdoor){
        if(txtTimeTaken != null){
            txtTimeTaken.setVisibility(View.INVISIBLE);
        }
        timeStarted = System.nanoTime();

        if(sharedPreferences != null){
            boolean storedBigTurn = sharedPreferences.getBoolean(RobotControllerActions.PREF_BIG_TURN, false);
            boolean storedOutdoor = sharedPreferences.getBoolean(RobotControllerActions.PREF_OUTDOOR, false);
            if(storedBigTurn != isBigTurn || storedOutdoor != isOutdoor){
                sharedPreferences.edit()
                        .putBoolean(RobotControllerActions.PREF_BIG_TURN, isBigTurn)
                        .putBoolean(RobotControllerActions.PREF_OUTDOOR, isOutdoor)
                        .apply();
            }
        }

        if(turningModeSwitch != null && turningModeSwitch.isChecked() != isBigTurn){
            turningModeSwitch.setChecked(isBigTurn);
        }

        if(outdoorArenaSwitch != null && outdoorArenaSwitch.isChecked() != isOutdoor){
            outdoorArenaSwitch.setChecked(isOutdoor);
        }

        if(isBigTurn){
            if(isOutdoor){
                sendTurningModeCmdIntent("WN04");
            }else{
                sendTurningModeCmdIntent("WN02");
            }
        }else{
            if(isOutdoor){
                sendTurningModeCmdIntent("WN03");
            }else{
                sendTurningModeCmdIntent("WN01");
            }
        }
    }

    private BroadcastReceiver roboStatusUpdateReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            try{
                String msgInfo = intent.getStringExtra("msg");
                txtRoboStatus.setText(msgInfo);
            }catch (Exception e){
                txtRoboStatus.setText("UNKNOWN");
                showShortToast("Error updating robocar status");
                Log.e(TAG, "onReceive: An error occured while updating the robocar status");
                e.printStackTrace();
            }
        }
    };

    private BroadcastReceiver roboStateReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            try{
                String state = intent.getStringExtra("msg");
                switch(state.toUpperCase()){
                    case "FINISHED":
                        timeEnded = System.nanoTime();
                        timeTakenInNanoSeconds = timeEnded - timeStarted;

                        double timeTakenInSeconds = (double) timeTakenInNanoSeconds/1000000000;
                        int timeTakenMin = (int) timeTakenInSeconds/60;
                        double timeTakenSec = (double) timeTakenInSeconds%60;
                        DecimalFormat df = new DecimalFormat("0.00");

                        txtTimeTaken.setText("Run completed in: "+Integer.toString(timeTakenMin)+"min "+df.format(timeTakenSec)+"secs");
                        txtTimeTaken.setVisibility(View.VISIBLE);

                        btnSetObstacle.setEnabled(true);
                        btnPlaceRobot.setEnabled(true);
                        btnResetArena.setEnabled(true);
                        btnSetFacing.setEnabled(true);
                        btnSendStartFastestCar.setEnabled(true);
                        btnSendStartImageRec.setEnabled(true);
                        btnSendArenaInfo.setEnabled(true);
                        btnAddObsManual.setEnabled(true);
                        break;
                    case "RUNNING":
                        btnSetObstacle.setEnabled(false);
                        btnPlaceRobot.setEnabled(false);
                        btnResetArena.setEnabled(false);
                        btnSetFacing.setEnabled(false);
                        btnSendStartFastestCar.setEnabled(false);
                        btnSendStartImageRec.setEnabled(false);
                        btnSendArenaInfo.setEnabled(false);
                        btnAddObsManual.setEnabled(false);
                        break;
                }
            }catch (Exception ex){
                Log.e(TAG, "onReceive: Error receiving robot completion status");
            }
        }
    };

    private BroadcastReceiver roboModeUpdateReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            try{
                String mode = intent.getStringExtra("msg");
                switch (mode.toUpperCase()){
                    case "PATH":
                        manualModeSwitch.setChecked(false);
                        break;
                    case "MANUAL":
                        manualModeSwitch.setChecked(true);
                        break;
                }
            }catch (Exception ex){
                Log.e(TAG, "onReceive: An error occured on receiving robocar mode");
                ex.printStackTrace();
            }
        }
    };

    private final BroadcastReceiver startFastestCarReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            boolean isBigTurn = intent.getBooleanExtra(RobotControllerActions.EXTRA_BIG_TURN,
                    turningModeSwitch != null && turningModeSwitch.isChecked());
            boolean isOutdoor = intent.getBooleanExtra(RobotControllerActions.EXTRA_OUTDOOR,
                    outdoorArenaSwitch != null && outdoorArenaSwitch.isChecked());
            startFastestCar(isBigTurn, isOutdoor);
        }
    };

    private BroadcastReceiver updateObstalceListReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            obstacleListItemList.clear();
            try{
                JSONArray msgInfo = new JSONArray(intent.getStringExtra("msg"));
                for(int i=0; i<msgInfo.length();i++){
                    JSONObject obj = msgInfo.getJSONObject(i);
                    obstacleListItemList.add(new ObstacleListItem(obj.getInt("no"), obj.getInt("x"),obj.getInt("y"),obj.getString("facing")));
                }
                obstaclesListViewAdapter.updateList(obstacleListItemList);
                refreshObstacleListHeight();
            }catch (Exception ex){
                Log.e(TAG, "onReceive: An error occured while updating obstacle list view");
                ex.printStackTrace();
            }
        }
    };

    private BroadcastReceiver robotLocationUpdateReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            try{
                JSONObject msgJSON = new JSONObject(intent.getStringExtra("msg"));
                int xCoord = msgJSON.getInt("x");
                int yCoord = msgJSON.getInt("y");
                int dirInt = msgJSON.getInt("d");
                GridMap.Direction direction = GridMap.Direction.UP;
                switch(dirInt){
                    case 0: //NORTH
                        direction = GridMap.Direction.UP;
                        break;
                    case 2: //EAST
                        direction = GridMap.Direction.RIGHT;
                        break;
                    case 4: //SOUTH
                        direction = GridMap.Direction.DOWN;
                        break;
                    case 6: //WEST
                        direction = GridMap.Direction.LEFT;
                        break;
                }

                if(xCoord < 0 || yCoord < 0 || xCoord > 20 || yCoord > 20){
                    showShortToast("Error: Robot move out of area (x: "+xCoord+", y: "+yCoord+")");
                    Log.e(TAG, "onReceive: Robot is out of the arena area");
                    return;
                }

                if (gridMap != null) {
                    gridMap.updateCurCoord(xCoord, yCoord, direction);
                }
            }catch (Exception e){
                showShortToast("Error updating robot location");
                Log.e(TAG, "onReceive: An error occured while updating robot location");
                e.printStackTrace();
            }
        }
    };

    private BroadcastReceiver imageRecResultReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            try{
                JSONObject msgJSON = new JSONObject(intent.getStringExtra("msg"));
                int obstacleID = Integer.parseInt(msgJSON.getString("obstacle_id"));
                String targetID = msgJSON.getString("image_id");
                if (gridMap != null) {
                    gridMap.updateImageNumberCell(obstacleID, targetID);
                }
            }catch (Exception e){
                showShortToast("Error updating image rec result");
                Log.e(TAG, "onReceive: An error occured while upating the image rec result");
                e.printStackTrace();
            }
        }
    };

    private void showShortToast(String msg) {
        Toast.makeText(getActivity(), msg, Toast.LENGTH_SHORT).show();
    }

    private void showLongToast(String msg) {
        Toast.makeText(getActivity(), msg, Toast.LENGTH_LONG).show();
    }

    private void attachManualControlListener(ImageButton button, String forwardCommand) {
        button.setOnTouchListener((v, event) -> {
            int action = event.getAction();
            if (action == MotionEvent.ACTION_DOWN) {
                v.setPressed(true);
                sendDirectionCmdIntent(forwardCommand);
                return true;
            }

            if (action == MotionEvent.ACTION_UP) {
                v.setPressed(false);
                sendDirectionCmdIntent("STOP");
                v.performClick();
                return true;
            }

            if (action == MotionEvent.ACTION_CANCEL) {
                v.setPressed(false);
                sendDirectionCmdIntent("STOP");
                return true;
            }

            return false;
        });
    }

    private void sendDirectionCmdIntent(String direction){

        try{
            JSONObject directionJSONObj = new JSONObject();
            directionJSONObj.put("cat","manual");
            directionJSONObj.put("value",direction);

            broadcastSendBTIntent(directionJSONObj.toString());
        }catch (Exception e){
            Log.e(TAG, "sendDirectionCmdIntent: An error occured while sending direction command intent");
            e.printStackTrace();
        }
    }

    private void sendModeCmdIntent(String mode){
        try{
            if(!mode.equals("path") && !mode.equals("manual")){
                Log.i(TAG, "sendModeIntent: Invalid mode to send: "+mode);
                return;
            }
            JSONObject modeJSONObj = new JSONObject();
            modeJSONObj.put("cat","mode");
            modeJSONObj.put("value",mode);

            broadcastSendBTIntent(modeJSONObj.toString());
        }catch (Exception e){
            Log.e(TAG, "sendModeIntent: An error occured while sending mode command intent");
            e.printStackTrace();
        }
    }

    private void sendTurningModeCmdIntent(String mode){
        try{
            JSONObject modeJSONObj = new JSONObject();
            modeJSONObj.put("cat","manual");
            modeJSONObj.put("value",mode);

            broadcastSendBTIntent(modeJSONObj.toString());
        }catch (Exception e){
            Log.e(TAG, "sendModeIntent: An error occured while sending mode command intent");
            e.printStackTrace();
        }
    }

    private void sendControlCmdIntent(String control){
        try{
            JSONObject ctrlJSONObj = new JSONObject();
            ctrlJSONObj.put("cat","control");
            ctrlJSONObj.put("value",control);

            broadcastSendBTIntent(ctrlJSONObj.toString());
        }catch (Exception e){
            Log.e(TAG, "sendControlCmdIntent: An error occured while sending control command intent");
            e.printStackTrace();
        }
    }

    private void broadcastSendBTIntent(String msg){
        Intent sendBTIntent = new Intent("sendBTMessage");
        sendBTIntent.putExtra("msg",msg);
        LocalBroadcastManager.getInstance(getContext()).sendBroadcast(sendBTIntent);
    }

    private void refreshObstacleListHeight() {
        if (obstacleListView == null) {
            return;
        }

        obstacleListView.post(() -> {
            ListAdapter adapter = obstacleListView.getAdapter();
            if (adapter == null) {
                return;
            }

            int totalHeight = 0;
            int widthMeasureSpec = View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED);
            for (int i = 0; i < adapter.getCount(); i++) {
                View listItem = adapter.getView(i, null, obstacleListView);
                listItem.measure(widthMeasureSpec, View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED));
                totalHeight += listItem.getMeasuredHeight();
            }

            int dividerHeight = obstacleListView.getDividerHeight();
            if (dividerHeight < 0) {
                dividerHeight = 0;
            }
            totalHeight += dividerHeight * Math.max(adapter.getCount() - 1, 0);

            ViewGroup.LayoutParams params = obstacleListView.getLayoutParams();
            params.height = totalHeight;
            obstacleListView.setLayoutParams(params);
            obstacleListView.requestLayout();
        });
    }

    private class ObstaclesListViewAdapter extends ArrayAdapter<ObstacleListItem>{
        private List<ObstacleListItem> items;

        public ObstaclesListViewAdapter(@NonNull Context context, int resource, @NonNull List<ObstacleListItem> objects) {
            super(context, resource, objects);
            items=objects;
        }

        public void updateList(List<ObstacleListItem> list) {
            this.items = list;
            this.notifyDataSetChanged();
        }

        @NonNull
        @Override
        public View getView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {
            if (convertView == null) {
                convertView = LayoutInflater.from(getContext()).inflate(R.layout.home_obstacle_list_layout, parent, false);
            }
            ObstacleListItem item = items.get(position);
            TextView obsNoTxt = (TextView) convertView.findViewById(R.id.txtObsListItem_obsNo);
            TextView coordinatesTxt = (TextView) convertView.findViewById(R.id.txtObsListItem_coordinates);
            TextView facingTxt = (TextView) convertView.findViewById(R.id.txtObsListItem_dir);

            Locale locale = Locale.getDefault();
            obsNoTxt.setText(String.format(locale, "Obstacle #%02d", item.obsNo));
            coordinatesTxt.setText(String.format(locale, "Coordinates: (%02d, %02d)", item.x, item.y));

            String facingValue = item.facing == null || item.facing.trim().isEmpty()
                    ? "--"
                    : item.facing;
            facingTxt.setText(facingValue.toUpperCase(locale));

            return convertView;
        }
    }

    private class ObstacleListItem {
        int obsNo;
        int x;
        int y;
        String facing;

        public ObstacleListItem(int obsNo,int x, int y, String facing){
            this.obsNo = obsNo;
            this.x=x;
            this.y=y;
            this.facing=facing;
        }
    }
}



