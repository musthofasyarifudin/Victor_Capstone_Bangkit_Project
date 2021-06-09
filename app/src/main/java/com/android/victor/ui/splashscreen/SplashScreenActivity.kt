package com.android.victor.ui.splashscreen

import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.WindowInsets
import android.view.WindowManager
import com.android.victor.databinding.ActivitySplashScreenBinding
import com.android.victor.ui.home.HomeActivity
import com.android.victor.ui.intro.IntroActivity
import com.android.victor.utils.Constants.KEY_EMAIL
import com.android.victor.utils.PreferenceManager
import com.android.victor.utils.Utils.start

@Suppress("DEPRECATION")
class SplashScreenActivity : AppCompatActivity() {

    private lateinit var b: ActivitySplashScreenBinding
    private lateinit var pref: PreferenceManager

    companion object {
        private const val TIME_LENGTH = 3000
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivitySplashScreenBinding.inflate(layoutInflater)
        setContentView(b.root)
        setFullscreen()
        initiate()
        goToApp()
    }

    private fun setFullscreen() {
        //Set fullscreen layout
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.insetsController?.hide(WindowInsets.Type.statusBars())
        } else {
            window.setFlags(
                WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN
            )
        }
    }

    private fun initiate() {
        pref = PreferenceManager(this)
    }

    private fun goToApp() {
        Handler(Looper.getMainLooper()).postDelayed({
            if (pref.getEmail(KEY_EMAIL) != null){
                start(this, HomeActivity::class.java)
            } else {
                start(this, IntroActivity::class.java)
            }
        }, TIME_LENGTH.toLong())
    }
}