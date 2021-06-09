package com.android.victor.ui.home

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import com.android.victor.R
import com.android.victor.api.DaggerApiComponent
import com.android.victor.api.UsersApi
import com.android.victor.databinding.ActivityHomeBinding
import com.android.victor.model.Data.generateDataAdditional
import com.android.victor.model.Data.generateDataBody
import com.android.victor.model.Data.generateDataCategory
import com.android.victor.model.Data.generateDataDigestion
import com.android.victor.model.Data.generateDataEyes
import com.android.victor.model.Data.generateDataMental
import com.android.victor.model.Data.generateDataSensation
import com.android.victor.model.Data.generateDataSkin
import com.android.victor.model.Data.generateDataTht
import com.android.victor.model.MessageModel
import com.android.victor.model.PredictResponse
import com.android.victor.model.Symptoms
import com.android.victor.utils.Utils.showToast
import org.json.JSONArray
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.util.*
import javax.inject.Inject
import kotlin.collections.ArrayList


class HomeActivity : AppCompatActivity(), MessageAdapter.OnItemClick, View.OnClickListener {

    @Inject
    lateinit var usersApi: UsersApi
    init {
        DaggerApiComponent.create().inject(this)
    }

    private var items: ArrayList<String> = arrayListOf()

    private lateinit var b: ActivityHomeBinding
    private lateinit var messageAdapter: MessageAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityHomeBinding.inflate(layoutInflater)
        setContentView(b.root)
        initiate()
        setupRecyclerView()
    }

    private fun initiate() {
        b.btnLogout.setOnClickListener(this)
        messageAdapter = MessageAdapter(arrayListOf())
    }

    private fun setupRecyclerView() {
        messageAdapter.addSingleMessage(
            MessageModel("Hello, my name is Victor, and I will help you to find out your disease. Tell me what your symptoms are. Do you want to start?", textLength = "Long")
        )
        messageAdapter.addSingleMessage(
            MessageModel("Start", messageId = "start")
        )
        messageAdapter.addSingleMessage(
            MessageModel("Close", messageId = "close")
        )
        b.rvMessage.apply {
            adapter = messageAdapter
            layoutManager = LinearLayoutManager(this@HomeActivity).apply {
                stackFromEnd = true
            }
        }
        messageAdapter.listener = this
    }

    override fun onMessageClick(message: MessageModel) {
        when(message.messageId){
            "start" -> showCategories(true)
            "close" -> close()
            "finish" -> checkout()
            "back" -> showCategories(false)
            "body" -> categories(message)
            "digestion" -> categories(message)
            "eyes" -> categories(message)
            "mental" -> categories(message)
            "sensation" -> categories(message)
            "skin" -> categories(message)
            "tht" -> categories(message)
            "additional" -> categories(message)
            else -> addItems(message)
        }
    }

    private fun showCategories(isFirstTime: Boolean) {
        Handler(Looper.getMainLooper()).postDelayed({
            if (isFirstTime){
                messageAdapter.addSingleMessage(MessageModel(messageText = "Choose which part has symptoms of the disease?", textLength = "Long"))
            } else {
                messageAdapter.addSingleMessage(MessageModel(messageText = "Is there anything else?", textLength = "Long"))
            }
            messageAdapter.addItemMessage(generateDataCategory())
            smoothScroll()
        }, 1000)
    }

    private fun close() {
        this.finishAffinity()
    }

    private fun categories(message: MessageModel) {
        //Add user chat
        messageAdapter.addSingleMessage(MessageModel(message.messageText, "user"))
        smoothScroll()
        //Feedback user chat
        Handler(Looper.getMainLooper()).postDelayed({
            if (message.messageId == "additional"){
                messageAdapter.addSingleMessage(
                    MessageModel("What are your complaints about your body?", "victor", textLength = "Long")
                )
            } else {
                messageAdapter.addSingleMessage(
                    MessageModel("What are your complaints about your ${message.messageId}?", "victor", textLength = "Long")
                )
            }
            smoothScroll()
        }, 1000)
        //Show detail per categories
        Handler(Looper.getMainLooper()).postDelayed({
            when(message.messageId){
                "body" -> messageAdapter.addItemMessage(generateDataBody())
                "digestion" -> messageAdapter.addItemMessage(generateDataDigestion())
                "eyes" -> messageAdapter.addItemMessage(generateDataEyes())
                "mental" -> messageAdapter.addItemMessage(generateDataMental())
                "sensation" -> messageAdapter.addItemMessage(generateDataSensation())
                "skin" -> messageAdapter.addItemMessage(generateDataSkin())
                "tht" -> messageAdapter.addItemMessage(generateDataTht())
                "additional" -> messageAdapter.addItemMessage(generateDataAdditional())
            }
            smoothScroll()
        }, 3000)
    }

    private fun addItems(message: MessageModel) {
        if (items.size <= 7){
            messageAdapter.addSingleMessage(MessageModel(message.messageText, "user"))
            smoothScroll()
            items.add(message.messageId)
            Log.e("TAG", "DATA : ${items.size}")
            Handler(Looper.getMainLooper()).postDelayed({
                showCategories(false)
            }, 1000)
        } else {
            Toast.makeText(this, "You have given enough symptoms. So, click finish button below!", Toast.LENGTH_LONG).show()
            showCategories(false)
        }
    }

    private fun checkout() {
        if (items.size != 0){
            while(items.size < 17){
                items.add("0")
            }
            postData()
        } else {
            Toast.makeText(this, "Choose at least one symptoms!", Toast.LENGTH_LONG).show()
        }
    }

    private fun postData() {
        val data = JSONArray(items)
        val symptoms = Symptoms(items)
        Log.e("TAG","RESPONSE: $items")
        Log.e("TAG","RESPONSE: $data")
        usersApi.predict(symptoms).enqueue(object : Callback<PredictResponse>{
            override fun onResponse(
                call: Call<PredictResponse>,
                response: Response<PredictResponse>
            ) {
                if (response.isSuccessful) {
                    messageAdapter.addSingleMessage(
                        MessageModel("Victor's Prediction: ")
                    )
                    messageAdapter.addSingleMessage(
                        MessageModel("${response.body()?.prediction} \n ${response.body()?.desc}", textLength = "Long")
                    )
                    messageAdapter.addSingleMessage(
                        MessageModel("Precautions: ")
                    )
                    for (i in response.body()?.precautions!!){
                        messageAdapter.addSingleMessage(
                            MessageModel("- $i", textLength = "Long")
                        )
                    }
                    smoothScroll()
                } else {
                    showToast(this@HomeActivity, "The server is having problems, try again later")
                }

                Log.e("TAG","RESPONSE: ${response.code()}")
                Log.e("TAG","RESPONSE: ${response.message()}")
            }

            override fun onFailure(call: Call<PredictResponse>, t: Throwable) {
                showToast(this@HomeActivity, "The server is having problems, try again later")
            }
        })
    }

    private fun smoothScroll() {
        b.rvMessage.smoothScrollToPosition(messageAdapter.countItemMessage())
    }

    override fun onClick(v: View?) {
        if (v?.id == R.id.btnLogout){
            val dialog = LogoutDialog()
            dialog.show(supportFragmentManager, "DIALOG")
        }
    }
}