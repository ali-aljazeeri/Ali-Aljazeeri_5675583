import AdminLayout from "@/Layouts/AdminLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/Components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/Components/ui/table";
import { Badge } from "@/Components/ui/badge";
import { Button } from "@/Components/ui/button";
import { Eye } from "lucide-react";
import { Link } from "@inertiajs/react";

export default function Damages({ crashes }) {
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

    return (
        <AdminLayout
            title="Damage Reports Admin"
            description="View and manage all submitted damage reports"
        >
            <Card>
                <CardHeader>
                    <CardTitle>Damage Reports</CardTitle>
                    <CardDescription>
                        View and manage all submitted damage reports
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>ID</TableHead>
                                <TableHead>Vehicle Number</TableHead>
                                <TableHead>Email</TableHead>
                                <TableHead>Status</TableHead>
                                <TableHead>Submitted At</TableHead>
                                <TableHead>Actions</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {crashes.map((crash) => (
                                <TableRow key={crash.id}>
                                    <TableCell>{crash.id}</TableCell>
                                    <TableCell>{crash.vehicle_number}</TableCell>
                                    <TableCell>{crash.email}</TableCell>
                                    <TableCell>
                                        <Badge variant={getStatusBadgeVariant(crash.status)}>
                                            {crash.status.charAt(0).toUpperCase() + crash.status.slice(1)}
                                        </Badge>
                                    </TableCell>
                                    <TableCell>
                                        {new Date(crash.created_at).toLocaleString()}
                                    </TableCell>
                                    <TableCell>
                                        <Button
                                            variant="outline"
                                            size="sm"
                                            asChild
                                        >
                                            <Link href={route('crashes.show', crash.id)}>
                                                <Eye className="h-4 w-4 mr-1"/>
                                                View
                                            </Link>
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>
        </AdminLayout>
    );
}
