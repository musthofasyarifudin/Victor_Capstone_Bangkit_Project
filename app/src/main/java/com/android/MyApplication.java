package com.android;

import android.app.Application;
import android.content.Context;

public class MyApplication extends Application {

    private static MyApplication instance;

    @Override
    public void onCreate() {
        super.onCreate();
        instance = this;
    }

    public static MyApplication getInstance() { return instance; }

    @Override
    protected void attachBaseContext(Context base) {
        super.attachBaseContext(base);
    }
}
