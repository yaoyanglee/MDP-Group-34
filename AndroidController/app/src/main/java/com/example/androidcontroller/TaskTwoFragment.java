package com.example.androidcontroller;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

public class TaskTwoFragment extends Fragment {

    private boolean isStarted = false; // toggle state tracker

    public TaskTwoFragment() {
        // Required empty public constructor
    }

    public static TaskTwoFragment newInstance() {
        return new TaskTwoFragment();
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_task_two, container, false);

        ImageButton startButton = view.findViewById(R.id.btnTaskTwoStartFastestCar);

        startButton.setOnClickListener(v -> {
            Context context = requireContext();
            SharedPreferences preferences = context.getSharedPreferences(
                    RobotControllerActions.PREFS_NAME, Context.MODE_PRIVATE);

            boolean isBigTurn = preferences.getBoolean(RobotControllerActions.PREF_BIG_TURN, false);
            boolean isOutdoor = preferences.getBoolean(RobotControllerActions.PREF_OUTDOOR, false);

            // Toggle start/stop state
            isStarted = !isStarted;
            startButton.setSelected(isStarted); // changes the image (selector uses state_selected)

            if (isStarted) {
                // Send broadcast to start fastest car
                Intent intent = new Intent(RobotControllerActions.ACTION_START_FASTEST_CAR);
                intent.putExtra(RobotControllerActions.EXTRA_BIG_TURN, isBigTurn);
                intent.putExtra(RobotControllerActions.EXTRA_OUTDOOR, isOutdoor);
                LocalBroadcastManager.getInstance(context).sendBroadcast(intent);

                Toast.makeText(context,
                        buildLaunchMessage(isBigTurn, isOutdoor),
                        Toast.LENGTH_SHORT).show();

            } else {
                // Optional: you can send a "stop" broadcast or just show a toast
                Toast.makeText(context, "Fastest Car stopped.", Toast.LENGTH_SHORT).show();
            }
        });

        return view;
    }

    private String buildLaunchMessage(boolean isBigTurn, boolean isOutdoor) {
        String turnMode = isBigTurn ? "Big Turn" : "Normal Turn";
        String arenaMode = isOutdoor ? "Outdoor" : "Indoor";
        return "Starting Fastest Car (" + turnMode + " • " + arenaMode + ")";
    }
}
