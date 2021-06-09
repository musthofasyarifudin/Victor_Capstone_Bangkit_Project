package com.android.victor.ui.register

import android.content.Intent
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import com.android.victor.R
import com.android.victor.databinding.ActivityRegisterBinding
import com.android.victor.ui.intro.IntroActivity
import com.android.victor.utils.Utils.hideLoading
import com.android.victor.utils.Utils.showLoading
import com.android.victor.utils.Utils.showMessage
import com.android.victor.utils.Utils.showToast
import com.android.victor.utils.Utils.start
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.FirebaseUser
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.storage.StorageReference
import kotlin.random.Random

class RegisterActivity: AppCompatActivity(), View.OnClickListener {

    private var fullName: String? = null
    private var email: String? = null
    private var phone: String? = null
    private var password: String? = null
    private var passwordConfirm: String? = null

    private lateinit var b : ActivityRegisterBinding
    private lateinit var registerViewModel: RegisterViewModel
    private lateinit var auth: FirebaseAuth
    private lateinit var firestore: FirebaseFirestore

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityRegisterBinding.inflate(layoutInflater)
        setContentView(b.root)
        initiate()
        setClickListener()
        checkData()
    }

    private fun initiate() {
        auth = FirebaseAuth.getInstance()
        firestore = FirebaseFirestore.getInstance()
        registerViewModel = ViewModelProvider(this).get(RegisterViewModel::class.java)
    }

    private fun setClickListener() {
        b.btnRegister.setOnClickListener(this)
    }

    private fun checkData() {
        b.btnRegister.isEnabled = false
        b.etEmail.addTextChangedListener(textWatcher)
        b.etFullName.addTextChangedListener(textWatcher)
        b.etPhone.addTextChangedListener(textWatcher)
        b.etPassword.addTextChangedListener(textWatcher)
        b.etPasswordConfirm.addTextChangedListener(textWatcher)
    }

    private val textWatcher = object : TextWatcher {
        override fun afterTextChanged(s: Editable?) {}
        override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
        override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
            b.btnRegister.isEnabled =
                        b.etEmail.text.toString().isNotEmpty() &&
                        b.etFullName.text.toString().isNotEmpty() &&
                        b.etPhone.text.toString().isNotEmpty() &&
                        b.etPassword.text.toString().isNotEmpty() &&
                        b.etPasswordConfirm.text.toString().isNotEmpty()
            fullName = b.etFullName.text.toString()
            email = b.etEmail.text.toString()
            phone = b.etPhone.text.toString()
            password = b.etPassword.text.toString()
            passwordConfirm = b.etPasswordConfirm.text.toString()
        }
    }

    private fun isValidated() : Boolean {
        return when {
            password?.length!! < 8 -> {
                showMessage(b.root, "Your password is less than 8 characters")
                false
            }
            password != passwordConfirm -> {
                showMessage(b.root, "Make sure the password you enter is correct")
                false
            }
            else -> {
                true
            }
        }
    }

    private fun register() {
        registerViewModel.checkEmailInUser(email!!, password!!, fullName!!, phone!!)
        registerViewModel.loading.observe(this, { isLoading ->
            if(isLoading == true) {
                showLoading(this, b.progressBar.progressbar)
            } else {
                hideLoading(this, b.progressBar.progressbar)
            }
        })
        registerViewModel.successRegister.observe(this, { isSuccess ->
            if (isSuccess == true) {
                showToast(this, getString(R.string.register_succes))
                start(this, IntroActivity::class.java)
            }
        })
        registerViewModel.errorMessage.observe(this, { errorMessage ->
            registerViewModel.showMessage.observe(this, { showMessage ->
                if (showMessage == true) {
                    showMessage(b.root, errorMessage)
                }
            })
        })
    }

    override fun onClick(v: View?) {
        when(v?.id){
            R.id.btnRegister -> if(isValidated()) register()
        }
    }
}