package com.example.omu

interface Platform {
    val name: String
}

expect fun getPlatform(): Platform