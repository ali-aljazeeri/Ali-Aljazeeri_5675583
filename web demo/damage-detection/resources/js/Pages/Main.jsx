import { Button } from "@/Components/ui/button";
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "@/Components/ui/card";
import { Input } from "@/Components/ui/input";
import { Label } from "@/Components/ui/label";
import { useForm, usePage } from "@inertiajs/react";
import { useState, useEffect } from "react";
import { Camera, AlertCircle } from "lucide-react";
import FlashMessage from "@/Components/FlashMessage";

export default function Main() {
    const { flash = {} } = usePage().props;
    const [isMobileDevice, setIsMobileDevice] = useState(true);
    const { data, setData, post, processing, errors } = useForm({
        vehicleNumber: '',
        email: '',
        image1: null,
        image2: null,
        image3: null,
        image4: null
    });

    const [previews, setPreviews] = useState({
        image1: null,
        image2: null,
        image3: null,
        image4: null
    });

    useEffect(() => {
        // Check if the device has camera capabilities
        const checkDevice = () => {
            const hasTouchScreen = navigator.maxTouchPoints > 0;
            const isMobile = /Mobi|Android/i.test(navigator.userAgent);

            setIsMobileDevice(hasTouchScreen && isMobile);
            // setIsMobileDevice(true);
        };

        checkDevice();
        window.addEventListener('resize', checkDevice);

        return () => window.removeEventListener('resize', checkDevice);
    }, []);

    const handleSubmit = (e) => {
        e.preventDefault();
        post(route('damages.store'), {
            preserveScroll: true,
        });
    };

    const handleImageCapture = (e, imageKey) => {
        const file = e.target.files[0];
        if (file) {
            setData(imageKey, file);
            // Create preview URL
            setPreviews(prev => ({
                ...prev,
                [imageKey]: URL.createObjectURL(file)
            }));
        }
    };

    return (
        <div className="min-h-screen bg-gray-100">
            {/* Header */}
            <header className="bg-white shadow-sm">
                <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
                    <h1 className="text-2xl font-bold text-gray-900">
                        Car Damage Report System
                    </h1>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
                {!isMobileDevice ? (
                    <Card className="max-w-md mx-auto">
                        <CardContent className="pt-6">
                            <div className="flex items-center gap-4 text-destructive">
                                <AlertCircle className="h-10 w-10" />
                                <div className="space-y-2">
                                    <h2 className="text-lg font-semibold">Desktop Access Not Supported</h2>
                                    <p className="text-sm">
                                        This application requires a mobile device with a camera. Please open this page on your smartphone/tablet.
                                    </p>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                ) : (
                    <Card className="max-w-md mx-auto">
                        <CardHeader>
                            <CardTitle>Submit Damage Report</CardTitle>
                            <CardDescription>
                                Please fill in the details about the incident
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-4">
                                {/* Flash Messages */}
                                {flash.success && <FlashMessage type="success" message={flash.success} />}
                                {flash.error && <FlashMessage type="error" message={flash.error} />}
                                {flash.info && <FlashMessage type="info" message={flash.info} />}
                                {flash.warning && <FlashMessage type="warning" message={flash.warning} />}

                                <form onSubmit={handleSubmit} className="space-y-4">
                                    <div className="space-y-2">
                                        <Label htmlFor="vehicleNumber">Vehicle Number</Label>
                                        <Input
                                            id="vehicleNumber"
                                            type="text"
                                            placeholder="Enter vehicle number"
                                            value={data.vehicleNumber}
                                            onChange={(e) => setData('vehicleNumber', e.target.value)}
                                            required
                                        />
                                        {errors.vehicleNumber && (
                                            <p className="text-sm text-red-500">{errors.vehicleNumber}</p>
                                        )}
                                    </div>

                                    <div className="space-y-2">
                                        <Label htmlFor="email">Email</Label>
                                        <Input
                                            id="email"
                                            type="email"
                                            placeholder="Enter your email"
                                            value={data.email}
                                            onChange={(e) => setData('email', e.target.value)}
                                            required
                                        />
                                        {errors.email && (
                                            <p className="text-sm text-red-500">{errors.email}</p>
                                        )}
                                    </div>

                                    <div className="space-y-4">
                                        <Label className="text-lg font-medium">Capture Images</Label>
                                        <p className="text-sm text-gray-500">Please take 4 photos of the side</p>

                                        {/* Image 1 */}
                                        <div className="space-y-2">
                                            <Label>Front View</Label>
                                            <div className="flex flex-col items-center gap-4">
                                                <Input
                                                    id="image1"
                                                    type="file"
                                                    accept="image/*"
                                                    capture="environment"
                                                    className="hidden"
                                                    onChange={(e) => handleImageCapture(e, 'image1')}
                                                    required
                                                />
                                                <Button
                                                    type="button"
                                                    variant="outline"
                                                    className="w-full"
                                                    onClick={() => document.getElementById('image1').click()}
                                                >
                                                    <Camera className="mr-2 h-4 w-4" />
                                                    Take Front View
                                                </Button>

                                                {previews.image1 && (
                                                    <div className="relative w-full aspect-video rounded-lg overflow-hidden">
                                                        <img
                                                            src={previews.image1}
                                                            alt="Front View Preview"
                                                            className="object-cover w-full h-full"
                                                        />
                                                    </div>
                                                )}
                                            </div>
                                            {errors.image1 && (
                                                <p className="text-sm text-red-500">{errors.image1}</p>
                                            )}
                                        </div>

                                        {/* Image 2 */}
                                        <div className="space-y-2">
                                            <Label>Rear View</Label>
                                            <div className="flex flex-col items-center gap-4">
                                                <Input
                                                    id="image2"
                                                    type="file"
                                                    accept="image/*"
                                                    capture="environment"
                                                    className="hidden"
                                                    onChange={(e) => handleImageCapture(e, 'image2')}
                                                    required
                                                />
                                                <Button
                                                    type="button"
                                                    variant="outline"
                                                    className="w-full"
                                                    onClick={() => document.getElementById('image2').click()}
                                                >
                                                    <Camera className="mr-2 h-4 w-4" />
                                                    Take Rear View
                                                </Button>

                                                {previews.image2 && (
                                                    <div className="relative w-full aspect-video rounded-lg overflow-hidden">
                                                        <img
                                                            src={previews.image2}
                                                            alt="Side View Preview"
                                                            className="object-cover w-full h-full"
                                                        />
                                                    </div>
                                                )}
                                            </div>
                                            {errors.image2 && (
                                                <p className="text-sm text-red-500">{errors.image2}</p>
                                            )}
                                        </div>

                                        {/* Image 3 */}
                                        <div className="space-y-2">
                                            <Label>Right View</Label>
                                            <div className="flex flex-col items-center gap-4">
                                                <Input
                                                    id="image3"
                                                    type="file"
                                                    accept="image/*"
                                                    capture="environment"
                                                    className="hidden"
                                                    onChange={(e) => handleImageCapture(e, 'image3')}
                                                    required
                                                />
                                                <Button
                                                    type="button"
                                                    variant="outline"
                                                    className="w-full"
                                                    onClick={() => document.getElementById('image3').click()}
                                                >
                                                    <Camera className="mr-2 h-4 w-4" />
                                                    Take Right View
                                                </Button>

                                                {previews.image3 && (
                                                    <div className="relative w-full aspect-video rounded-lg overflow-hidden">
                                                        <img
                                                            src={previews.image3}
                                                            alt="Rear View Preview"
                                                            className="object-cover w-full h-full"
                                                        />
                                                    </div>
                                                )}
                                            </div>
                                            {errors.image3 && (
                                                <p className="text-sm text-red-500">{errors.image3}</p>
                                            )}
                                        </div>

                                        {/* Image 4 */}
                                        <div className="space-y-2">
                                            <Label>Left View</Label>
                                            <div className="flex flex-col items-center gap-4">
                                                <Input
                                                    id="image4"
                                                    type="file"
                                                    accept="image/*"
                                                    capture="environment"
                                                    className="hidden"
                                                    onChange={(e) => handleImageCapture(e, 'image4')}
                                                    required
                                                />
                                                <Button
                                                    type="button"
                                                    variant="outline"
                                                    className="w-full"
                                                    onClick={() => document.getElementById('image4').click()}
                                                >
                                                    <Camera className="mr-2 h-4 w-4" />
                                                    Take Left View
                                                </Button>

                                                {previews.image4 && (
                                                    <div className="relative w-full aspect-video rounded-lg overflow-hidden">
                                                        <img
                                                            src={previews.image4}
                                                            alt="Damage Close-up Preview"
                                                            className="object-cover w-full h-full"
                                                        />
                                                    </div>
                                                )}
                                            </div>
                                            {errors.image4 && (
                                                <p className="text-sm text-red-500">{errors.image4}</p>
                                            )}
                                        </div>
                                    </div>

                                    <Button type="submit" className="w-full" disabled={processing}>
                                        {processing ? 'Submitting...' : 'Submit Report'}
                                    </Button>
                                </form>
                            </div>
                        </CardContent>
                    </Card>
                )}
            </main>
        </div>
    );
}
