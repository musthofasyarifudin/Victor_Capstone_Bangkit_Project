package com.android.victor.ui.login

import android.content.Intent
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import com.android.victor.R
import com.android.victor.databinding.ActivityLoginBinding
import com.android.victor.model.User
import com.android.victor.ui.home.HomeActivity
import com.android.victor.ui.register.RegisterActivity
import com.android.victor.utils.PreferenceManager
import com.android.victor.utils.Utils.hideLoading
import com.android.victor.utils.Utils.showLoading
import com.android.victor.utils.Utils.showMessage
import com.android.victor.utils.Utils.start
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore

class LoginActivity: AppCompatActivity(), View.OnClickListener {

    private var email: String? = null
    private var password: String? = null
    private var user: User = User()

    private lateinit var loginViewModel: LoginViewModel
    private lateinit var auth: FirebaseAuth
    private lateinit var firestore: FirebaseFirestore
    private lateinit var b: ActivityLoginBinding
    private lateinit var pref: PreferenceManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityLoginBinding.inflate(layoutInflater)
        setContentView(b.root)
        initiate()
        setClickListener()
        checkData()
    }

    private fun initiate() {
        auth = FirebaseAuth.getInstance()
        firestore = FirebaseFirestore.getInstance()
        loginViewModel = ViewModelProvider(this).get(LoginViewModel::class.java)
        pref = PreferenceManager(this)
    }

    private fun setClickListener() {
        b.btnLogin.setOnClickListener(this)
        b.tvSignUp.setOnClickListener(this)
    }

    private fun checkData() {
        b.btnLogin.isEnabled = false
        b.etEmail.addTextChangedListener(textWatcher)
        b.etPassword.addTextChangedListener(textWatcher)
    }

    private val textWatcher = object : TextWatcher {
        override fun afterTextChanged(s: Editable?) {}
        override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
        override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
            b.btnLogin.isEnabled =
                b.etEmail.text.toString().isNotEmpty() && b.etPassword.text.toString().isNotEmpty()
            email = b.etEmail.text.toString()
            password = b.etPassword.text.toString()
        }
    }

    private fun checkEmailInUser() {
        loginViewModel.checkEmailInUser(email!!, password!!)
        loginViewModel.loading.observe(this, {
            if (it == true) {
                showLoading(this, b.progressBar.progressbar)
            } else {
                hideLoading(this, b.progressBar.progressbar)
            }
        })
        loginViewModel.user.observe(this, {
            user = it
        })
        loginViewModel.successSignIn.observe(this, { signIn ->
            if (signIn == true) {
                pref.savePersonalData(user)
                start(this, HomeActivity::class.java)
                finish()
            }
        })
        loginViewModel.errorMessage.observe(this, { errorMessage ->
            loginViewModel.showMessage.observe(this, { showMessage ->
                if (showMessage == true) {
                    showMessage(b.root, errorMessage)
                }
            })
        })

    }

    private fun register() {
        startActivity(Intent(this, RegisterActivity::class.java))
    }

    override fun onClick(v: View?) {
        when(v?.id){
            R.id.btnLogin -> checkEmailInUser()
            R.id.tvSignUp -> register()
        }
    }
}