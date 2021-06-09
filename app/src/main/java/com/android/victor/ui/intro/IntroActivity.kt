package com.android.victor.ui.intro

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import com.android.victor.R
import com.android.victor.databinding.ActivityIntroBinding
import com.android.victor.model.SlideModel
import com.android.victor.ui.login.LoginActivity
import com.android.victor.ui.register.RegisterActivity
import com.smarteist.autoimageslider.SliderAnimations
import com.smarteist.autoimageslider.SliderView

class IntroActivity: AppCompatActivity(), View.OnClickListener {

    private var itemSlider: ArrayList<SlideModel> = arrayListOf()

    private lateinit var adapter: IntroAdapter
    private lateinit var b: ActivityIntroBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityIntroBinding.inflate(layoutInflater)
        setContentView(b.root)
        initiate()
        addSliderItem()
        setClickListener()
    }

    private fun initiate() {
        adapter = IntroAdapter(arrayListOf())
    }

    private fun addSliderItem() {
        itemSlider.add(SlideModel(R.drawable.intro_image_1, resources.getString(R.string.title_slide_1), resources.getString(R.string.desc_slide_1)))
        itemSlider.add(SlideModel(R.drawable.intro_image_2, resources.getString(R.string.title_slide_2), resources.getString(R.string.desc_slide_2)))
        adapter.updateImage(itemSlider)
        setSliderView()
    }

    private fun setSliderView() {
        b.imageSlider.setSliderAdapter(adapter)
        b.imageSlider.setSliderTransformAnimation(SliderAnimations.SIMPLETRANSFORMATION)
        b.imageSlider.autoCycleDirection = SliderView.AUTO_CYCLE_DIRECTION_RIGHT
        b.imageSlider.indicatorSelectedColor = Color.DKGRAY
        b.imageSlider.indicatorUnselectedColor = Color.GRAY
        b.imageSlider.scrollTimeInSec = 3
        b.imageSlider.isAutoCycle = true
    }

    private fun setClickListener() {
        b.btnLogin.setOnClickListener(this)
        b.btnRegister.setOnClickListener(this)
    }

    private fun login() {
        val login = Intent(this, LoginActivity::class.java)
        startActivity(login)
    }

    private fun register() {
        val register = Intent(this, RegisterActivity::class.java)
        startActivity(register)
    }

    override fun onClick(v: View?) {
        when(v?.id){
            R.id.btn_login -> login()
            R.id.btn_register -> register()
        }
    }
}