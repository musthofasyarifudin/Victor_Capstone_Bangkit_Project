package com.android.victor.api

import com.android.MyApplication
import com.chuckerteam.chucker.api.ChuckerCollector
import com.chuckerteam.chucker.api.ChuckerInterceptor
import dagger.Module
import dagger.Provides
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.adapter.rxjava2.RxJava2CallAdapterFactory
import retrofit2.converter.gson.GsonConverterFactory

@Module
class ApiModule {

    var client: OkHttpClient = OkHttpClient.Builder()
        .addInterceptor(
            ChuckerInterceptor.Builder(MyApplication.getInstance())
                .collector(ChuckerCollector(MyApplication.getInstance()))
                .maxContentLength(250000L)
                .redactHeaders(emptySet())
                .alwaysReadResponseBody(false)
                .build()
        )
        .build()

    @Provides
    fun provideUsersApi(): UsersApi {
        return Retrofit.Builder()
            .baseUrl("http://34.87.166.51/")
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .addCallAdapterFactory(RxJava2CallAdapterFactory.create())
            .build()
            .create(UsersApi::class.java)
    }
}