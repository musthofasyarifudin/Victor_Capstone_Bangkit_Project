package com.android.victor.utils

import android.content.Context
import com.android.victor.model.User

class PreferenceManager(context: Context) {
    private val preferences = context.getSharedPreferences("User_Pref", Context.MODE_PRIVATE)

    fun savePersonalData(user: User){
        setEmail(Constants.KEY_EMAIL, user.email)
        setName(Constants.KEY_FULL_NAME, user.name)
        setPhone(Constants.KEY_PHONE, user.phone)
    }

    fun setName(key: String, value: String){
        preferences.edit().putString(key, value).apply()
    }

    fun getName(key: String): String?{
        return preferences.getString(key, null)
    }

    fun setEmail(key: String, value: String){
        preferences.edit().putString(key, value).apply()
    }

    fun getEmail(key: String): String?{
        return preferences.getString(key, null)
    }

    fun setPhone(key: String, value: String){
        preferences.edit().putString(key, value).apply()
    }

    fun getPhone(key: String): String?{
        return preferences.getString(key, null)
    }

    fun clearUserCredentials() {
        preferences.edit().clear().apply()
    }
}