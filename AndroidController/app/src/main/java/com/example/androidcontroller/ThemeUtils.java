package com.example.androidcontroller;

import android.content.Context;
import android.content.SharedPreferences;

import androidx.appcompat.app.AppCompatDelegate;

/**
 * Utility class for applying and persisting the selected theme mode across the app.
 */
public final class ThemeUtils {

    private ThemeUtils() {
    }

    private static SharedPreferences getPreferences(Context context) {
        return context.getSharedPreferences(RobotControllerActions.PREFS_NAME, Context.MODE_PRIVATE);
    }

    public static boolean isDarkModeEnabled(Context context) {
        return getPreferences(context).getBoolean(RobotControllerActions.PREF_DARK_MODE, true);
    }

    public static void setDarkMode(Context context, boolean enabled) {
        getPreferences(context).edit().putBoolean(RobotControllerActions.PREF_DARK_MODE, enabled).apply();
        AppCompatDelegate.setDefaultNightMode(enabled ? AppCompatDelegate.MODE_NIGHT_YES : AppCompatDelegate.MODE_NIGHT_NO);
    }

    public static void applyTheme(Context context) {
        AppCompatDelegate.setDefaultNightMode(isDarkModeEnabled(context)
                ? AppCompatDelegate.MODE_NIGHT_YES
                : AppCompatDelegate.MODE_NIGHT_NO);
    }
}
