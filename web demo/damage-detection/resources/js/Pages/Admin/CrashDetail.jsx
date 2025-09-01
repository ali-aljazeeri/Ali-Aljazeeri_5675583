import AdminLayout from "@/Layouts/AdminLayout";
import {Card, CardContent, CardDescription, CardHeader, CardTitle} from "@/Components/ui/card";
import {Badge} from "@/Components/ui/badge";
import {Button} from "@/Components/ui/button";
import {ArrowLeft, RefreshCw} from "lucide-react";
import {Link, useForm} from "@inertiajs/react";

export default function CrashDetail({crash}) {

    const reprocessForm = useForm();

    const handleReprocess = () => {
        reprocessForm.post(route('damages.reprocess', crash.id), {
            onSuccess: () => {
                // Success message will be handled by the backend redirect
            },
        });
    };

    const getStatusBadgeVariant = (status) => {
        switch (status) {
            case 'approved':
                return 'success';
            case 'rejected':
                return 'destructive';
            default:
                return 'secondary';
        }
    };

    const getSuccessFailureBadge = (isSuccess) => {
        return (
            <Badge variant={isSuccess ? 'success' : 'destructive'}>
                {isSuccess ? 'Success' : 'Failure'}
            </Badge>
        );
    };

    const renderPredictionResults = (imageKey, imageData) => {
        const runs = imageData.runs;
        const isPredict1Success = runs.predict1?.vehicle_count === 1;
        const isPredict2Success = runs.predict2?.class === 1;
        const isPredict3Success = (runs.predict3?.cropped_vehicle_damage?.total_detections === 0 && runs.predict3?.full_image_damage?.total_detections === 0);

        return (
            <Card key={imageKey} className="mb-6">
                <CardHeader>
                    <CardTitle className="text-lg">{imageKey.charAt(0).toUpperCase() + imageKey.slice(1)}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                    {/* Original Image */}
                    <div>
                        <h4 className="font-medium text-gray-700 mb-2">Original Image</h4>
                        <div className="rounded-lg overflow-hidden border w-full flex justify-center bg-gray-50">
                            <img
                                src={`data:image/jpeg;base64,${imageData.file}`}
                                alt={`${imageKey} original`}
                                className="w-[300px] max-h-[400px] object-contain"
                            />
                        </div>
                    </div>

                    {/* Prediction Results */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                        {/* Predict 1 Results */}
                        <Card>
                            <CardHeader className="pb-3">
                                <CardTitle className="text-sm">Predict 1 - Vehicle Detection</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-3">
                                {runs.predict1 ? (
                                    <>
                                        {getSuccessFailureBadge(isPredict1Success)}
                                        {!isPredict1Success ? (<>
                                            <div>
                                                <span className="text-sm font-medium">Vehicle Count: </span>
                                                <span className="text-sm">{runs.predict1.vehicle_count}</span>
                                            </div>
                                            {runs.predict1.result_image_base64 && (
                                                <div>
                                                    <h5 className="text-sm font-medium mb-2">Result Image</h5>
                                                    <img
                                                        src={`data:image/jpeg;base64,${runs.predict1.result_image_base64}`}
                                                        alt="Predict 1 result"
                                                        className="w-full max-h-40 object-contain border rounded"
                                                    />
                                                </div>
                                            )}
                                        </>) : null}
                                    </>
                                ) : (
                                    <div className="text-gray-500 text-sm">Not processed</div>
                                )}
                            </CardContent>
                        </Card>

                        {/* Predict 2 Results */}
                        <Card>
                            <CardHeader className="pb-3">
                                <CardTitle className="text-sm">Predict 2 - Classification</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-3">
                                {runs.predict2 ? (
                                    <>
                                        {getSuccessFailureBadge(isPredict2Success)}
                                        {!isPredict2Success ? (<>
                                            <div>
                                                <span className="text-sm font-medium">Class: </span>
                                                <span className="text-sm">{runs.predict2.class}</span>
                                            </div>
                                            {runs.predict2.grad_cam_image_base64 && (
                                                <div>
                                                    <h5 className="text-sm font-medium mb-2">Grad-CAM Image</h5>
                                                    <img
                                                        src={`data:image/jpeg;base64,${runs.predict2.grad_cam_image_base64}`}
                                                        alt="Predict 2 grad-cam"
                                                        className="w-full max-h-40 object-contain border rounded"
                                                    />
                                                </div>
                                            )}
                                        </>) : null}
                                    </>
                                ) : (
                                    <div className="text-gray-500 text-sm">Not processed</div>
                                )}
                            </CardContent>
                        </Card>

                        {/* Predict 3 Results */}
                        <Card>
                            <CardHeader className="pb-3">
                                <CardTitle className="text-sm">Predict 3 - Damage Detection</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-3">
                                {runs.predict3 ? (
                                    <>
                                        {getSuccessFailureBadge(
                                            isPredict3Success
                                        )}

                                        {!isPredict3Success ? (<>
                                            <div>
                                                <span className="text-sm font-medium">Cropped total detections: </span>
                                                <span
                                                    className="text-sm">{runs.predict3.cropped_vehicle_damage?.total_detections}</span>
                                            </div>
                                            <div>
                                                <span className="text-sm font-medium">Full total detections: </span>
                                                <span
                                                    className="text-sm">{runs.predict3.full_image_damage?.total_detections}</span>
                                            </div>

                                            {/* Cropped Vehicle Damage */}
                                            {runs.predict3.cropped_vehicle_damage && (
                                                <div>
                                                    <h5 className="text-sm font-medium mb-1">Cropped Vehicle</h5>
                                                    <div className="text-xs mb-2">
                                                        Classes: {runs.predict3.cropped_vehicle_damage.detections?.map(d => d.class).join(', ') || 'None'}
                                                    </div>
                                                    {runs.predict3.cropped_vehicle_damage.visualization_base64 && (
                                                        <img
                                                            src={`data:image/jpeg;base64,${runs.predict3.cropped_vehicle_damage.visualization_base64}`}
                                                            alt="Cropped vehicle damage"
                                                            className="w-full max-h-32 object-contain border rounded mb-2"
                                                        />
                                                    )}
                                                </div>
                                            )}

                                            {/* Full Image Damage */}
                                            {runs.predict3.full_image_damage && (
                                                <div>
                                                    <h5 className="text-sm font-medium mb-1">Full Image</h5>
                                                    <div className="text-xs mb-2">
                                                        Classes: {runs.predict3.full_image_damage.detections?.map(d => d.class).join(', ') || 'None'}
                                                    </div>
                                                    {runs.predict3.full_image_damage.visualization_base64 && (
                                                        <img
                                                            src={`data:image/jpeg;base64,${runs.predict3.full_image_damage.visualization_base64}`}
                                                            alt="Full image damage"
                                                            className="w-full max-h-32 object-contain border rounded"
                                                        />
                                                    )}
                                                </div>
                                            )}
                                        </>) : null}
                                    </>
                                ) : (
                                    <div className="text-gray-500 text-sm">Not processed</div>
                                )}
                            </CardContent>
                        </Card>
                    </div>
                </CardContent>
            </Card>
        );
    };

    return (
        <AdminLayout
            title={
                <div className="flex items-center gap-4">
                    <Link href={route('damages.index')}>
                        <Button variant="outline" size="sm">
                            <ArrowLeft className="h-4 w-4 mr-1"/>
                            Back to List
                        </Button>
                    </Link>
                    <span>Damage Report Details</span>
                </div>
            }
        >
            <Card className="mb-6">
                <CardHeader className="flex flex-row items-center justify-between">
                    <div>
                        <CardTitle>Report #{crash.id}</CardTitle>
                        <CardDescription>
                            Submitted on {new Date(crash.created_at).toLocaleString()}
                        </CardDescription>
                    </div>
                    <Button
                        onClick={handleReprocess}
                        disabled={reprocessForm.processing}
                        variant="outline"
                        className="flex items-center gap-2"
                    >
                        <RefreshCw className={`h-4 w-4 ${reprocessForm.processing ? 'animate-spin' : ''}`}/>
                        {reprocessForm.processing ? 'Reprocessing...' : 'Reprocess'}
                    </Button>
                </CardHeader>
                <CardContent className="space-y-6">
                    <div className="grid grid-cols-3 gap-4">
                        <div>
                            <h3 className="font-medium text-gray-500">Vehicle Number</h3>
                            <p className="mt-1">{crash.vehicle_number}</p>
                        </div>
                        <div>
                            <h3 className="font-medium text-gray-500">Email</h3>
                            <p className="mt-1">{crash.email}</p>
                        </div>
                        <div>
                            <h3 className="font-medium text-gray-500">Status</h3>
                            <Badge variant={getStatusBadgeVariant(crash.status)} className="mt-1">
                                {crash.status.charAt(0).toUpperCase() + crash.status.slice(1)}
                            </Badge>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Images and Prediction Results */}
            <div className="space-y-6">
                <h2 className="text-xl font-semibold">Images and Prediction Results</h2>
                {crash.run_results && Object.keys(crash.run_results).map(imageKey =>
                    renderPredictionResults(imageKey, crash.run_results[imageKey])
                )}
            </div>
        </AdminLayout>
    );
}
