package com.android.victor.api

import com.android.victor.ui.home.HomeActivity
import dagger.Component

@Component(modules = [ApiModule::class])
interface ApiComponent {
    fun inject(homeActivity: HomeActivity)
}