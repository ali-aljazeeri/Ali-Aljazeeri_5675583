<?php

use Illuminate\Support\Facades\Route;
use Inertia\Inertia;
use App\Http\Controllers\DamageController;

Route::get('/', function () {
    return Inertia::render('Main');
})->name('home');

Route::post('/damages', [DamageController::class, 'store'])->name('damages.store');

Route::get('/success', function () {
    return Inertia::render('Success');
})->name('success');

// Admin routes
Route::get('/admin/damages', [DamageController::class, 'index'])->name('damages.index');
Route::get('/admin/damages/{crash}', [DamageController::class, 'show'])->name('crashes.show');
Route::post('/admin/damages/{crash}/reprocess', [DamageController::class, 'reprocess'])->name('damages.reprocess');
