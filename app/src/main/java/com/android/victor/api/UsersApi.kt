package com.android.victor.api

import com.android.victor.model.PredictResponse
import com.android.victor.model.Symptoms
import org.json.JSONArray
import retrofit2.Call
import retrofit2.http.Body
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface UsersApi {
    @POST("predict")
    fun predict(
        @Body data: Symptoms
    ): Call<PredictResponse>
}