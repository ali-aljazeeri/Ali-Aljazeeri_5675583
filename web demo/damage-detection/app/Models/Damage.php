<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Damage extends Model
{
    /**
     * The attributes that aren't mass assignable.
     *
     * @var array<string>
     */
    protected $guarded = [];
    protected $appends = ["status"];
    protected $casts = [
        'run_results' => 'json',
    ];

    public function getStatusAttribute(): string
    {
        $imageKeys = ['image1', 'image2', 'image3', 'image4'];

        // Check if any predict1 failed
        foreach ($imageKeys as $imageKey) {
            if (!isset($this->run_results[$imageKey]['runs']['predict1'])) {
                return 'rejected';
            }
            $predict1 = $this->run_results[$imageKey]['runs']['predict1'];
            if ($predict1['vehicle_count'] !== 1) {
                return 'rejected';
            }
        }

        // Check if any predict2 failed
        foreach ($imageKeys as $imageKey) {
            if (!isset($this->run_results[$imageKey]['runs']['predict2'])) {
                return 'rejected';
            }
            $predict2 = $this->run_results[$imageKey]['runs']['predict2'];
            if ($predict2['class'] === 0) {
                return 'rejected';
            }
        }

        // Check if any predict3 failed (but predict3 failure doesn't cause rejection)
        foreach ($imageKeys as $imageKey) {
            if (!isset($this->run_results[$imageKey]['runs']['predict3'])) {
                return 'rejected';
            }
            $predict3 = $this->run_results[$imageKey]['runs']['predict3'];
            $croppedDetections = $predict3['cropped_vehicle_damage']['total_detections'];
            $fullImageDetections = $predict3['full_image_damage']['total_detections'];

            if ($croppedDetections !== 0 || $fullImageDetections !== 0) {
                return 'rejected';
            }
        }
        // All predictions completed successfully
        return 'approved';
    }
}
