<?php

namespace App\Http\Controllers;

use App\Models\Damage;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Inertia\Inertia;

class DamageController extends Controller
{
    /**
     * Display a listing of crashes.
     */
    public function index()
    {
        $crashes = Damage::orderByDesc('created_at')->get();

        return Inertia::render('Admin/Damages', [
            'crashes' => $crashes->forget('runs')
        ]);
    }

    /**
     * Display the specified crash.
     */
    public function show(Damage $crash)
    {
        return Inertia::render('Admin/CrashDetail', [
            'crash' => $crash
        ]);
    }

//    /**
//     * Update the crash processed status.
//     */
//    public function updateStatus(Request $request, Crash $crash)
//    {
//        $validated = $request->validate([
//            'status' => ['required', 'in:pending,approved,rejected'],
//        ]);
//
//        dd($validated['status']);
//
//        $crash->update([
//            'status' => $validated['status'],
//        ]);
//
//
//        return back();
//    }

    /**
     * Store a newly created crash report in storage.
     */
    public function store(Request $request)
    {
        // Increase execution time limit to 5 minutes for API processing
        ini_set('max_execution_time', 300);

        // Check if the user agent is a mobile device
        $userAgent = $request->header('User-Agent');
        $isMobile = preg_match('/Mobi|Android/i', $userAgent);

        if (!$isMobile) {
            return Inertia::render('Errors/MobileOnly', [
                'message' => 'This application requires a mobile device with a camera.'
            ]);
        }
        $validated = $request->validate([
            'vehicleNumber' => 'required|string|max:255',
            'email' => 'required|email|max:255',
            'image1' => 'required|image|max:10240', // 10MB max
            'image2' => 'required|image|max:10240', // 10MB max
            'image3' => 'required|image|max:10240', // 10MB max
            'image4' => 'required|image|max:10240', // 10MB max
        ]);


        $run_results = [
            'image1' => [
                'file' => base64_encode(file_get_contents($request->file('image1')->getRealPath())),
                'runs' => [
                    'predict1' => null,
                    'predict2' => null,
                    'predict3' => null,
                ]
            ],
            'image2' => [
                'file' => base64_encode(file_get_contents($request->file('image2')->getRealPath())),
                'runs' => [
                    'predict1' => null,
                    'predict2' => null,
                    'predict3' => null,
                ]
            ],
            'image3' => [
                'file' => base64_encode(file_get_contents($request->file('image3')->getRealPath())),
                'runs' => [
                    'predict1' => null,
                    'predict2' => null,
                    'predict3' => null,
                ]
            ],
            'image4' => [
                'file' => base64_encode(file_get_contents($request->file('image4')->getRealPath())),
                'runs' => [
                    'predict1' => null,
                    'predict2' => null,
                    'predict3' => null,
                ]
            ]
        ];

        // Process the prediction workflow
        $this->processPredictionWorkflow($run_results);

        $crash = Damage::create([
            'vehicle_number' => $validated['vehicleNumber'],
            'email' => $validated['email'],
            'run_results' => $run_results,
        ]);

        return to_route('success');
    }

    /**
     * Process prediction workflow for all images
     */
    private function processPredictionWorkflow(&$run_results)
    {
        $imageKeys = ['image1', 'image2', 'image3', 'image4'];

        // Step 1: Run predict1 for all images
        $predict1Failed = false;
        foreach ($imageKeys as $imageKey) {
            $response = $this->callPredictAPI('predict1', $run_results[$imageKey]['file']);
            if ($response->failed()) {
                $predict1Failed = true;
                continue;
            }

            $run_results[$imageKey]['runs']['predict1'] = $response->json();

            if ($this->isPredict1Failed($response)) {
                $predict1Failed = true;
            }
        }

        // If any predict1 failed, stop the workflow
        if ($predict1Failed) {
            return;
        }

        // Step 2: Run predict2 for all images
        $predict2Failed = false;
        foreach ($imageKeys as $imageKey) {
            $response = $this->callPredictAPI('predict2', $run_results[$imageKey]['file']);
            if ($response->failed()) {
                $predict2Failed = true;
                continue;
            }

            $run_results[$imageKey]['runs']['predict2'] = $response->json();

            if ($this->isPredict2Failed($response)) {
                $predict2Failed = true;
            }
        }

        // If any predict2 failed, stop the workflow
        if ($predict2Failed) {
            return;
        }

        // Step 3: Run predict3 for all images
        foreach ($imageKeys as $imageKey) {
            $response = $this->callPredictAPI('predict3', $run_results[$imageKey]['file']);
            if ($response->failed()) {
                continue;
            }

            $run_results[$imageKey]['runs']['predict3'] = $response->json();

            // Note: We don't check for predict3 failure since it's the last step
        }
    }

    /**
     * Call prediction API for a specific endpoint and image
     */
    private function callPredictAPI($endpoint, $imageData)
    {
        $apiUrl = config('app.api_url');

        try {
            $response = Http::timeout(120)->post("{$apiUrl}/{$endpoint}", [
                'base64' => $imageData
            ]);

            return $response;
        } catch (\Exception $e) {
            // Return a failed response object if exception occurs
            return Http::response(['error' => $e->getMessage()], 500);
        }
    }

    /**
     * Check if predict1 API call failed based on response
     */
    private function isPredict1Failed($response)
    {
        // Check if the response JSON has "vehicle_count" equal to 1
        return $response->json('vehicle_count') !== 1;
    }

    /**
     * Check if predict2 API call failed based on response
     */
    private function isPredict2Failed($response)
    {
        return $response->json('class') === 0;
    }

    /**
     * Reprocess the crash data using the prediction workflow
     */
    public function reprocess(Damage $crash)
    {
        // Increase execution time limit to 5 minutes for API processing
        ini_set('max_execution_time', 300);

        // Get the existing run_results but reset the runs data
        $run_results = $crash->run_results;
        foreach ($run_results as $imageKey => $imageData) {
            $run_results[$imageKey]['runs'] = [
                'predict1' => null,
                'predict2' => null,
                'predict3' => null,
            ];
        }

        // Process the prediction workflow again
        $this->processPredictionWorkflow($run_results);

        // Update the crash record with new run_results
        $crash->update([
            'run_results' => $run_results,
        ]);

        return back()->with('info', 'Reprocessing done.');
    }

    /**
     * Check if predict3 API call failed based on response
     */
    private function isPredict3Failed($response)
    {
        $croppedDetections = $response->json('cropped_vehicle_damage.total_detections');
        $fullImageDetections = $response->json('full_image_damage.total_detections');

        return $croppedDetections !== 0 || $fullImageDetections !== 0;
    }
}

